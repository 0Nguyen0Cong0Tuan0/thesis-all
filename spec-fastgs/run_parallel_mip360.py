import os
import sys
import subprocess
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = "/kaggle/working/thesis-all"
DATA_ROOT = os.path.join(PROJECT_DIR, "spec-fastgs/datasets/mipnerf360")
IMAGES = "images_8"
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "spec-fastgs/output/mip360_images_8")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "DiBiay/spec-fastgs-mipneft360-images8"

SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
GPUS = ["0", "1"]

# Automatically configure CUDA paths
cuda_dirs = ["/usr/local/cuda", "/usr/local/cuda-12.2", "/usr/local/cuda-12.1", "/usr/local/cuda-12", "/usr/local/cuda-11.8", "/usr/local/cuda-11"]
detected_cuda = None
for d in cuda_dirs:
    if os.path.exists(os.path.join(d, "bin", "nvcc")):
        detected_cuda = d
        break
if detected_cuda:
    os.environ["CUDA_HOME"] = detected_cuda
    os.environ["PATH"] = f"{detected_cuda}/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{detected_cuda}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    print(f"✅ Auto-configured CUDA using path: {detected_cuda}")

# Initialize Hugging Face API
try:
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    print(f"✅ HuggingFace Dataset initialized: {REPO_ID}")
except Exception as e:
    print(f"⚠️ Error initializing HuggingFace repository: {e}")

def run_cmd(args, env, scene, log_file):
    cwd = os.path.join(PROJECT_DIR, "spec-fastgs")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- Running command: {' '.join(args)} ---\n")
        f.flush()
        res = subprocess.run(args, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            raise RuntimeError(f"Command failed with status {res.returncode}: {' '.join(args)}")

def train_scene(scene, gpu_id):
    source_path = os.path.join(DATA_ROOT, scene)
    model_path = os.path.join(OUTPUT_ROOT, scene)
    log_file = os.path.join(OUTPUT_ROOT, f"{scene}_training.log")
    
    os.makedirs(model_path, exist_ok=True)
    
    print(f"🚀 [GPU {gpu_id}] Bắt đầu huấn luyện scene '{scene}'...")
    
    # Configure env variables for subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Check if directories exist
    if not os.path.isdir(source_path):
        print(f"⚠️ [GPU {gpu_id}] Bỏ qua {scene}: không tìm thấy thư mục {source_path}")
        return False
        
    # Check resume capability
    if os.path.exists(os.path.join(model_path, "results.json")) or os.path.exists(os.path.join(model_path, "results_grouped.json")):
        print(f"⏩ [GPU {gpu_id}] Scene '{scene}' đã hoàn thành trước đó. Tiến hành upload lại...")
        upload_results(scene, model_path)
        return True
        
    start_time = time.time()
    try:
        # Step 1: extract_reflection_prior.py
        print(f"🔍 [GPU {gpu_id}] Scene '{scene}': Trích xuất reflection prior...")
        run_cmd([
            sys.executable, "extract_reflection_prior.py",
            "-s", source_path,
            "-i", IMAGES,
            "--ref_prior_method", "tan"
        ], env, scene, log_file)
        
        # Step 2: train.py
        print(f"🏋️ [GPU {gpu_id}] Scene '{scene}': Huấn luyện Spec-fastgs (30k iterations)...")
        run_cmd([
            sys.executable, "train.py",
            "-s", source_path,
            "-m", model_path,
            "-i", IMAGES,
            "--eval",
            "--iterations", "30000",
            "--densification_interval", "100",
            "--optimizer_type", "default",
            "--asg_degree", "64",
            "--is_real",
            "--is_indoor",
            "--sh_degree", "3",
            "--highfeature_lr", "0.02",
            "--grad_abs_thresh", "0.0004",
            "--specular_start_iter", "3000",
            "--ref_prior_method", "tan",
            "--sh_spec_grad_scale", "0.75",
            "--sh_spec_mask_start", "8000",
            "--sh_spec_mask_threshold", "0.75",
            "--sh_spec_min_metric_count", "2",
            "--use_ref_score",
            "--use_adaptive_prior",
            "--use_sh_spec_mask"
        ], env, scene, log_file)
        
        # Step 3: render.py
        print(f"🎬 [GPU {gpu_id}] Scene '{scene}': Kết xuất test views...")
        run_cmd([
            sys.executable, "render.py",
            "-m", model_path,
            "--skip_train"
        ], env, scene, log_file)
        
        # Step 4: metrics.py
        print(f"📊 [GPU {gpu_id}] Scene '{scene}': Đo đạc chỉ số (metrics)...")
        run_cmd([
            sys.executable, "metrics.py",
            "-m", model_path
        ], env, scene, log_file)
        
        elapsed = time.time() - start_time
        print(f"✅ [GPU {gpu_id}] Hoàn thành scene '{scene}' trong {elapsed/60:.2f} phút!")
        
        # Step 5: Upload results to HF
        upload_results(scene, model_path)
        return True
        
    except Exception as e:
        print(f"❌ [GPU {gpu_id}] Lỗi khi chạy scene '{scene}': {e}")
        # Log error to file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n❌ ERROR: {str(e)}\n")
        return False

def upload_results(scene, model_path):
    print(f"☁️ Đang upload kết quả của scene '{scene}' lên HuggingFace...")
    for attempt in range(3):
        try:
            api.upload_folder(
                folder_path=model_path,
                path_in_repo=f"mip360_images_8/{scene}",
                repo_id=REPO_ID,
                repo_type="dataset"
            )
            print(f"✅ Đã upload thành công scene '{scene}' lên HuggingFace!")
            break
        except Exception as err:
            print(f"⚠️ Thử lại lần {attempt+1}: Lỗi khi upload scene '{scene}': {err}")
            time.sleep(10)

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Initialize thread-safe GPU queue
    gpu_queue = queue.Queue()
    for g in GPUS:
        gpu_queue.put(g)
        
    def worker(scene):
        gpu_id = gpu_queue.get()
        try:
            success = train_scene(scene, gpu_id)
            return scene, success
        finally:
            gpu_queue.put(gpu_id)
            
    print(f"⚙️ Bắt đầu huấn luyện song song {len(SCENES)} scenes trên {len(GPUS)} GPUs...")
    print(f"GPUs khả dụng: {GPUS}")
    
    # Execute training in parallel
    with ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        futures = {executor.submit(worker, scene): scene for scene in SCENES}
        for future in as_completed(futures):
            scene = futures[future]
            try:
                scene, success = future.result()
                status_str = "thành công" if success else "thất bại"
                print(f"📢 Trạng thái scene '{scene}': {status_str}")
            except Exception as e:
                print(f"❌ Exception in scene '{scene}': {e}")

if __name__ == "__main__":
    main()
