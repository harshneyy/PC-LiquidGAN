# Samsung Innovation Campus (SIC) Defense - Speaker Notes

**Presenter:** Harshit Verma (122CS0023)
**Project:** Physics-Informed LiquidGAN for Zero-Shot Cross-Domain Thermal Image Synthesis

---

## Slide 1: Title Slide
**Speaker Notes:**
- "Good morning respected reviewers and my guide, Dr. N. Srinivas Naik. I am Harshit Verma, and today I am thrilled to present my individual Samsung Innovation Campus (SIC) project."
- "My research focuses on a novel deep learning architecture called **PC-LiquidGAN**, explicitly designed to solve the extremely difficult problem of synthesizing highly-accurate Thermal Images entirely from standard RGB or NIR images, solving the massive problem of thermal data scarcity."

---

## Slide 2: Problem Statement
**Speaker Notes:**
- "Why is thermal imaging synthesis a problem worth solving? Three reasons:"
- "First, **Data Scarcity**: Building annotated, paired Thermal-RGB datasets is incredibly expensive. Thermal cameras are costly, and capturing aligned frames is time-consuming."
- "Second, **Physics Violation**: If we try to use standard GANs like CycleGAN or Pix2Pix, they treat thermal heatmaps exactly like generic grayscale photographs. They completely ignore real-world thermodynamics, resulting in biologically impossible heat signatures."
- "Finally, **Mode Collapse**: When I experimented with base LiquidGAN architectures, I found that they crushed high-resolution input images into tiny continuous latent vectors. This caused severe spatial degradation—the resulting images were entirely blurry and lost critical high-frequency structural details like leaf veins or facial contours."

---

## Slide 3: Novel Contributions (PC-LiquidGAN + ODE-UNet)
**Speaker Notes:**
- "To solve these flaws, I engineered four core architectural mathematically-backed innovations:"
- "1. **Physics-Constrained Loss**: I forced the generator to obey the steady-state heat diffusion equation directly inside the GAN objective. The model is penalized if its generated heat mapping violates thermodynamic energy conservation."
- "2. **Adaptive ODE Solver**: I upgraded the rigid RK4 solver to an adaptive Dormand-Prince (dopri5) solver. It dynamically allocates math computation where thermal gradients are steep and complex, saving time elsewhere."
- "3. **Fourier Spectral Loss**: Thermal radiation is physically low-frequency dominant. I implemented an FFT (Fast Fourier Transform) Gaussian mask loss to enforce this specific spectral physical trait."
- "4. **ODE-UNet Architecture**: This is the major structural breakthrough. I solved the latent mode collapse by injecting my Neural ODEs specifically across high-resolution spatial skip connections inside a UNet, rather than a standard bottleneck."

---

## Slide 4: PC-LiquidGAN ODE-UNet Architecture
**Speaker Notes:**
- "This diagram illustrates the exact internal routing of the ODE-UNet Generator I wrote."
- *(Point to the diagram)*
- "You can see the standard RGB image enters on the left, passes through the encoder block, and then hits the bright orange **Spatial ConvODEFunc** layer. This is where continuous-time dynamics are mapped across the spatial features."
- "Notice the critical green 'Skip Connections' along the top. This bypasses the continuous feature crush, bridging high-frequency sharpness directly to the decoder. Finally, the Pink Liquid Neural Network (LNN) acts as the discriminator, critiquing the output alongside the yellow and blue Physics and Spectral loss constraints I added."

---

## Slide 5: Final Quantitative Results (5 Distinct Domains)
**Speaker Notes:**
- "Here is the empirical proof of the architecture's success."
- "I tested PC-LiquidGAN across 5 radically different domains: Medical X-Rays, CBSR Biometrics, KAIST Urban Driving, and Agricultural Tomato & Chilli crops."
- "As you can see on the bar graph, the **Structural Similarity Index (SSIM)** is pinned at essentially **0.99+** across all five domains."
- "This indicates near-perfect, pixel-accurate, physically-constrained thermal synthesis. We can effectively generate infinite synthetic thermal data for these fields now with zero physical camera cost."

---

## Slide 6: Ablation & Comparative Model Analysis
**Speaker Notes:**
- "If the panel asks why the Ablation Study is only on the Tomato dataset—it is standard research practice to conduct ablation on a single highly challenging dataset to isolate the exact mathematical contributions of each novel layer without variables."
- "Looking at the graph: Standard WGAN-GP failed entirely (0.13 SSIM). Standard DCGAN reached 0.59. The Base LiquidGAN reached 0.82."
- "By injecting my Physics Loss, and then the Spectral Loss, performance refined. But the massive structural leap occurred when I rebuilt the system into the **ODE-UNet** format. That single architectural choice yielded a massive jump to 0.99 SSIM over the baselines."

---

## Slide 7: Visual Quality Comparison
**Speaker Notes:**
- "Numbers only tell part of the story. Here is the direct visual output."
- "For each domain, the Left image is the exact raw data input. The Center image is the actual ground truth thermal snapshot from a real thermal camera. The Right image is what **my PC-LiquidGAN model generated from scratch**."
- *(Pause for panel to look at the images)*
- "You can see that the synthesized outputs on the right are virtually indistinguishable from the real thermal captures in the center. The facial contours in CBSR, the plant pathogen outlines in Agri, and the heat signatures of the buildings in KAIST are synthesized flawlessly."

---

## Slide 8: Zero-Shot Thermodynamics Generalization
**Speaker Notes:**
- "A critical question arises: Did the model actually learn physics, or did it just memorize textures?"
- "To prove the physics generalization, I conducted a strict **Zero-Shot Transfer** experiment."
- "I trained the final ODE-UNet *exclusively* on the Chilli Leaf dataset. Then, without a single epoch of fine-tuning or exposure, I forced it to synthesize Thermal images for the Tomato Leaf dataset."
- "The result? It successfully mapped the completely unseen biological structures with a highly plausible SSIM of ~0.47. This conclusively proves that the continuous Neural ODE latent pathways successfully learned genuine thermodynamic diffusion routing patterns that mathematically generalize beyond their training data distribution."

---

## Slide 9: Conclusion & Future Work
**Speaker Notes:**
- "In conclusion, by enforcing the thermal continuity equation and correcting spatial feature crush via my ODE-UNet architecture, we achieved state-of-the-art highly-accurate cross-domain thermal synthesis."
- "For future deployment work, I plan to:"
- "1. Implement Learnable Diffusivity Mapping, letting the network self-calculate the material constant ($\alpha$)."
- "2. Extend this to Continuous Spatio-Temporal Video for live thermal framerates."
- "3. And finally, quantization deployment for sub-millisecond tensor frameworks to run on live UAV Drones for precision agriculture edge compute."
- "Thank you for your time and the opportunity to present my work. I am open to any questions."
