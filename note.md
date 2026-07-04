Ok good job. Now can you teach me how Tan-Ikeuchi + TopHat detect the specular region?

Let me walk through this step by step, using the actual code from algo2_tan_ikeuchi_tophat in the notebook.

The physics: why a highlight elevates the minimum channel

The dichromatic reflection model says every pixel's color is a sum of two parts:

I(x) = diffuse_color(x) · shading(x)   +   specular_color(x) · highlight_intensity(x)

The diffuse term is colored (depends on the material — red apple, green plant). The specular term is usually close to the color of the illuminant — for ordinary white light, that means roughly equal amounts added to R, G, and B.

So take a pixel that's normally pure red diffuse: (R, G, B) ≈ (0.9, 0.05, 0.05). Its minimum channel Imin = min(R,G,B) ≈ 0.05 — near zero, because nothing is pushing the low channels up.

Now put a highlight on that same red surface: the white specular light adannels: (1.0, 0.45, 0.45) (R clips at 1.0). Now Imin ≈ 0.45 — it jumpedfrom 0.05 to 0.45.

That's the whole idea: Imin(x) stays near 0 for ordinary colored diffuse lly where a whitish highlight has been added. That's Tan & Ikeuchi's"specular-free image" insight — Imin acts as a per-pixel proxy for "how much white light got added here."

Why raw Imin thresholding fails

The problem: Imin is also naturally high on any surface that's bright and already close to white/gray — a painted wall, a window, a white ceramic plate — even with zero specular contribution. A white wall might be (0.85, 0.83, 0.87); Imin ≈ 0.83, indistinguishable from a real highlight, because per-pixel color statistics alone can't tell "this
pixel is white because the whole wall is painted white" from "this pixel landed on it."

This is a spatial distinction, not a color one — so a per-pixel test stru's exactly what killed the ungated version (15-18% of the image flagged,whole walls/windows lit up red).

The fix: morphological white top-hat

This is the key trick, and it's a spatial-shape argument, not a color arg

opened = grey_opening(Imin, footprint=_disk_footprint(tophat_radius))
tophat = np.clip(Imin - opened, 0.0, None)

grey_opening with a disk of radius r asks, at every pixel: "what's the highest floor I can find if I slide a disk of radius r around this neighborhood, always keeping the
disk's minimum-covered value?" Concretely, opening = erosion (shrink brign in the disk) followed by dilation (grow back by taking local max in thedisk). The net effect: any bright feature narrower than the disk gets erased, while large bright regions survive basically unchanged.

- On a flat white wall: every pixel's neighborhood looks the same, so opened ≈ Imin there. tophat = Imin - opened ≈ 0. The wall contributes nothing.
- On a small highlight glint (say 5px wide, radius=12 disk): the disk is bigger than the glint, so at the glint's center the opening operation can't "protect" that bright blob — it gets flattened to the level of the surrounding (darker) pixels. openImin - opened is large. The glint survives.

So tophat isolates compact bright blobs narrower than the disk, while suppressing anything broad and flat — regardless of whether that flat thing is bright or dim. This is why it doesn't matter if the background is black or white (I verified this directly on the teapot images): top-hat measures local elevation above the neighborhood floor, not
absolute brightness.

Final mask

mask = (tophat > tophat_thresh) & (Imax > bright_floor)

Two conditions, both must hold:
1. tophat > tophat_thresh — this pixel is a compact bright blob (the spat
2. Imax > bright_floor — the pixel is also bright in absolute terms (rules out, e.g., a small dark object against an even darker background — small ≠ specular by itself)

The limitation we actually found                                                                                                                                              
This gate is a real fix for wide flat surfaces, but it's not the same mechanism the real Tan-Ikeuchi paper uses (which iteratively compares local diffuse chromaticity, not a one-shot morphological filter). The gap shows up specifically on large tedow has internal structure (frame lines, glare gradients, glimpses of anoutdoor scene) that's still "bumpy" in the Imin channel even though the whole window is saturated in Imax. Top-hat can't tell "bumpy because it's textured glass" from "bumpy because it's a real highlight," which is why gated Tan-Ikeuchi still overwindow-containing images — the thing that ultimately kept it out ofproduction.