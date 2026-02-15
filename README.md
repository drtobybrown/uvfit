# uvfit

**High-performance 3D spectral-line fitting to interferometric visibilities.**

`uvfit` fits 3D models (x, y, v) directly to u-v visibilities, avoiding the
artifacts and noise correlations introduced by imaging. Inspired by
[galario](https://github.com/mtazzari/galario), but extended to full 3D
spectral-line cubes (e.g., ALMA CO measurement sets).

## Features

- **Visibility-space fitting** — compare models to data in the Fourier domain
- **Differentiable pipeline** — supports gradient-based optimization via JAX
- **Flexible models** — use an existing image cube as a template, or generate
  cubes from physical parameters with [KinMS](https://kinms.space)
- **GPU-accelerated** — optional JAX/PyTorch backends for production workloads
- **xradio I/O** — reads measurement sets via the
  [xradio](https://xradio.readthedocs.io) MSv4 schema

## Quick Start

```bash
# Install (CPU-only, for development/testing)
pip install -e ".[dev]"

# Run tests
pytest

# Install with JAX GPU support
pip install -e ".[jax,dev]"
```

### Minimum Working Example

```python
import numpy as np
from uvfit import UVDataset, TemplateCubeModel, Fitter

# Create a toy Gaussian cube
cube = ...  # shape (n_chan, ny, nx)

# Mock visibility data
uvdata = UVDataset.from_mock(u=u, v=v, vis=vis, weights=weights, freqs=freqs)

# Set up a template model and fit
model = TemplateCubeModel(template_cube=cube, cell_size=0.5)
fitter = Fitter(uvdata=uvdata, forward_model=model)
result = fitter.fit(initial_params={"dx": 0.1, "dy": -0.1, "flux_scale": 1.1})

print(result)
```

## Architecture and approach

```
Parameters → ForwardModel → 3D Cube → NUFFT → Model Visibilities → χ² → Optimizer
                                         ↑
                                    UVDataset (u, v, freqs)
```

**Pipeline:** A **ForwardModel** turns parameters into a 3D image cube `(n_chan, ny, nx)`. The cube is transformed to the Fourier domain via a **NUFFT** and sampled at the observed (u, v) and frequencies; the result is compared to the **UVDataset** with a χ² likelihood. An **Optimizer** (e.g. L-BFGS-B or emcee) updates the parameters. All comparison is done in visibility space, so the data are never imaged and the fit is not affected by imaging artifacts or noise correlations.

**Models:** Two main paths. (1) **TemplateCubeModel** — shift and scale a reference cube (dx, dy, dv, flux_scale); good for registration and flux scaling when the template is given. (2) **KinMSModel** (or custom ForwardModel) — map **physical** parameters (inclination, PA, velocity profile, etc.) to a cube, then to visibilities; use this to infer physical parameters directly from visibilities without circularity.

**Backends:** The NUFFT and gradient computation can run on CPU (NumPy/SciPy), or on GPU via optional JAX/PyTorch backends for faster fitting.

### How TemplateCubeModel works

`TemplateCubeModel` builds the model cube from a **reference data cube** (the template) by applying four operations controlled by the fit parameters. It does not re-simulate physics; it only shifts and scales the template so that the same pipeline (cube → NUFFT → visibilities) can be compared to the observed visibilities.

**Inputs**

- **`template_cube`**: 3D array of shape `(n_chan, ny, nx)` — the reference cube (e.g. a noisy observation or a KinMS cube). Stored internally as float64.
- **`cell_size_arcsec`**: Pixel size in arcseconds, used to convert sky shifts (`dx`, `dy`) into pixel shifts.
- **`channel_width_kms`** (optional): Channel width in km/s for velocity; default 1.0. Used for bounds and interpretation; the shift `dv` is in **channels**, not km/s.

**Parameters**

| Parameter     | Meaning | Default |
|--------------|---------|---------|
| `dx`         | Spatial shift in x (arcseconds) | 0 |
| `dy`         | Spatial shift in y (arcseconds) | 0 |
| `dv`         | Velocity shift in **channels** (fractional allowed) | 0 |
| `flux_scale` | Multiplicative flux scaling | 1.0 |

**How the model cube is created (`generate_cube`)**

1. **Parameter lookup**  
   `dx`, `dy`, `dv`, and `flux_scale` are read from the params dict (defaults 0, 0, 0, 1).

2. **Sky to pixel conversion**  
   Spatial shifts are converted to pixel shifts:  
   `dx_pix = dx / cell_size_arcsec`,  
   `dy_pix = dy / cell_size_arcsec`.  
   So a positive `dx` moves the template to the right (positive x in pixel space).

3. **3D sub-pixel shift**  
   The template is shifted with `scipy.ndimage.shift` in the order **channel, then y, then x**:
   - **`shift=[dv, dy_pix, dx_pix]`** — first axis is channel (velocity), second is y, third is x.
   - **`order=3`** — cubic interpolation for sub-pixel accuracy.
   - **`mode='constant', cval=0.0`** — outside the original grid the cube is filled with zero.

   So the model is the **same template**, shifted in position and velocity with no resizing or convolution.

4. **Flux scaling**  
   The shifted cube is multiplied by `flux_scale`. So the output is  
   `model_cube = flux_scale * shift(template_cube; dx, dy, dv)`.

**Result**

- Output is a 3D float64 array of shape `(n_chan, ny, nx)`, in the same grid as the template.  
- The fitter then uses this cube with the NUFFT (and optional backends) to compute model visibilities and χ² against the `UVDataset`.  
- Typical use: template = observed or simulated cube; fit `(dx, dy, dv, flux_scale)` so that the shifted, scaled template best matches the visibility data in the Fourier domain.

### Comparison with galario

| | galario | uvfit |
|---|--------|--------|
| **Domain** | 2D image (continuum or moment) | 3D cube (x, y, velocity) |
| **Transform** | FFT of 2D image, sample at (u,v) | NUFFT of 3D cube (per channel / spectral line) |
| **Typical use** | Disks, axisymmetric profiles, continuum | Full spectral-line cubes (e.g. CO, H I) |
| **Models** | Built-in profiles, arbitrary 2D images, translation/rotation | Template (shift/scale) or physical (e.g. KinMS) |

uvfit is inspired by galario’s visibility-domain fitting but is built for **spectral-line** data: the model is a 3D cube and the comparison to data is done in (u, v, channel) space, so you can fit kinematics and line structure without ever imaging.

### Deriving physical parameters in UV space — is it circular?

**Short answer:** You can derive physical parameters in UV space without circularity **if** the model is a **forward model** from physics to visibilities. That is the case when you use **KinMSModel** (or any `ForwardModel` that maps physical parameters → cube → visibilities). It is **not** the case when you only use **TemplateCubeModel** with a template that was built from the same data.

**Two ways to use uvfit:**

1. **TemplateCubeModel (template = some cube)**  
   - You fit only **geometric/flux** parameters: `dx`, `dy`, `dv`, `flux_scale`.  
   - You do **not** fit inclination, PA, velocity curve, surface brightness, etc.  
   - If the template is “the same observation, imaged,” you are effectively aligning and scaling that image in the Fourier domain. That gives you a proper UV-domain comparison and robust shift/scale, but no new *physical* parameters — the “physics” is whatever was already in the template. So it’s **not circular** in the sense of wrong math, but the only parameters you infer are position, velocity offset, and flux scale.  
   - If the template is **independent** (e.g. another line, a simulation, or a prior model), then you are genuinely fitting that model to the data in UV space; still only 4 parameters, but no circularity.

2. **KinMSModel (or another physical forward model)**  
   - Parameters are **physical**: e.g. inclination, position angle, flux, systemic velocity, gas dispersion, plus surface-brightness and velocity profiles.  
   - Pipeline: **physical params → KinMS (or similar) → 3D cube → NUFFT → model visibilities → χ² vs. data.**  
   - The data are never imaged to build the model; the model is fully predictive from the parameters. So you **are** deriving physical parameters directly in UV space, and it is **not circular**.  
   - This is the right path if the goal is to infer inclination, PA, rotation curve, etc., from visibilities.

**Summary:** For **physical parameter inference** in UV space, use a forward model (e.g. KinMSModel) so that parameters → cube → visibilities; that is non-circular. TemplateCubeModel is for alignment and flux scaling of a given cube; if that cube is the imaged data, you are not inferring new physics, only shift and scale.

## License

MIT
