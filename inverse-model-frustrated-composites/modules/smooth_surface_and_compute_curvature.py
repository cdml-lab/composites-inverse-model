# ┌───────────────────────────────────────────────────────────────────────────┐
# │                        Surface Smoothing & Curvature                     │
# └───────────────────────────────────────────────────────────────────────────┘

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for stable plot generation
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.signal import savgol_filter
import cv2  # OpenCV for bilateral filter (requires opencv-python)
from skimage.restoration import denoise_tv_chambolle
from scipy.interpolate import griddata

# Define sample indices to debug specific points
SAMPLE_INDICES = [0, 2, 33, 34, 201, 176, 9, 100]  # <--- manually set these indices

def smooth_savgol(Z, sigma):
    window = int(2 * sigma + 1)
    if window % 2 == 0:
        window += 1
    Z_smooth = savgol_filter(Z, window_length=window, polyorder=2, axis=0)
    Z_smooth = savgol_filter(Z_smooth, window_length=window, polyorder=2, axis=1)
    return Z_smooth

def smooth_bilateral(Z, sigma):
    # Normalize and convert to 8-bit for OpenCV
    Z_norm = (Z - np.min(Z)) / (np.max(Z) - np.min(Z)) * 255
    Z_8bit = np.uint8(Z_norm)
    smoothed = cv2.bilateralFilter(Z_8bit, d=9, sigmaColor=75, sigmaSpace=sigma)
    return smoothed.astype(float) / 255 * (np.max(Z) - np.min(Z)) + np.min(Z)


def rebuild_and_resample_surface(X, Y, Z, coarse_shape):
    """
    Rebuilds the surface using a coarse UV grid and then resamples it back
    to the original X, Y grid. If cubic interpolation produces NaNs, falls
    back to linear interpolation.

    Parameters:
    - X, Y, Z: 2D arrays of shape (nx, ny) — original surface.
    - coarse_shape: tuple (u, v) — size of the coarse grid to build smoothing surface.

    Returns:
    - Z_smooth: 2D array of shape (nx, ny) — smoothed version of original Z.
    """
    nx, ny = X.shape
    u, v = coarse_shape

    # Step 1: Create a coarse UV grid
    Xc = np.linspace(X.min(), X.max(), u)
    Yc = np.linspace(Y.min(), Y.max(), v)
    Xc_grid, Yc_grid = np.meshgrid(Xc, Yc, indexing='ij')

    # Interpolate original Z to this coarse grid
    original_points = np.column_stack((X.flatten(order='F'), Y.flatten(order='F')))
    Z_values = Z.flatten(order='F')
    Zc = griddata(original_points, Z_values, (Xc_grid, Yc_grid), method='cubic')

    # Fallback to linear if cubic returns NaNs
    if np.isnan(Zc).any():
        Zc = griddata(original_points, Z_values, (Xc_grid, Yc_grid), method='linear')

    # Step 2: Interpolate back to original resolution
    coarse_points = np.column_stack((Xc_grid.flatten(order='F'), Yc_grid.flatten(order='F')))
    Zc_values = Zc.flatten(order='F')
    Z_smooth = griddata(coarse_points, Zc_values, (X, Y), method='cubic')

    if np.isnan(Z_smooth).any():
        Z_smooth = griddata(coarse_points, Zc_values, (X, Y), method='linear')

    return Z_smooth
def smooth_surface_and_compute_curvature(base_dir, input_files_list, grid_shape,
                                         smoothing_method='gaussian', sigma=1.0,
                                         suffix="smooth"):

    """
    Processes Excel files containing 3D surface data arranged as a regular grid (Fortran order).
    For each sheet:
        - Reshapes and smooths the Z-values using Gaussian filtering
        - Computes principal curvatures using finite differences
        - Overwrites the corresponding curvature columns
        - Saves the new Excel file with '_smooth_curvature' suffix
        - Saves debug images for first 3 sheets

    Parameters:
    - base_dir: Base directory path (string or Path)
    - input_files_list: List of Excel file names (relative to base_dir)
    - grid_shape: tuple (nx, ny) describing the 2D grid dimensions
    """

    input_files_list = [Path(base_dir) / file for file in input_files_list]

    for file_path in input_files_list:
        # print(f"Processing: {file_path.name}")
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        name_parts = file_path.stem.split('_')

        if name_parts[-1].isdigit():
            new_name = '_'.join(name_parts[:-1]) + f"_{suffix}_{name_parts[-1]}.xlsx"
        else:
            new_name = file_path.stem + f"_{suffix}.xlsx"
        output_path = file_path.with_name(new_name)


        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

        debug_output_dir = file_path.parent / "debug_plots"
        debug_output_dir.mkdir(exist_ok=True)

        for i, sheet_name in enumerate(sheet_names):
            # print(f"  Sheet: {sheet_name}")
            df = pd.read_excel(xls, sheet_name=sheet_name)

            required_cols = ['Location X', 'Location Y', 'Location Z']
            if not all(col in df.columns for col in required_cols):
                # print(f"  Skipping {sheet_name} (missing XYZ columns)")
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                continue

            try:
                # Validate size
                expected_size = grid_shape[0] * grid_shape[1]
                if len(df) != expected_size:
                    raise ValueError(f"Expected {expected_size} points for shape {grid_shape}, got {len(df)}")

                # DEBUGGING MATCHED POINTS
                # for idx in SAMPLE_INDICES:
                #     if 0 <= idx < len(df):
                #         print("Sample index", idx)
                #         print("Original XYZ:", df.loc[idx, ['Location X', 'Location Y', 'Location Z']].values)

                # Reshape into 2D grid
                nx, ny = grid_shape
                X = df['Location X'].to_numpy().reshape((nx, ny), order='F')
                Y = df['Location Y'].to_numpy().reshape((nx, ny), order='F')
                Z = df['Location Z'].to_numpy().reshape((nx, ny), order='F')

                dx = dy = 1.0  # cm spacing
                X = df['Location X'].to_numpy().reshape((nx, ny), order='F')
                Y = df['Location Y'].to_numpy().reshape((nx, ny), order='F')
                Z = df['Location Z'].to_numpy().reshape((nx, ny), order='F')

                # print("    Reshaped XYZ into grid")

                # Smooth Z using selected method
                if smoothing_method == 'gaussian':
                    Z_smooth = gaussian_filter(Z, sigma=sigma)
                    # print("    Applied Gaussian smoothing")
                elif smoothing_method == 'uniform':
                    Z_smooth = uniform_filter(Z, size=int(2 * sigma + 1))
                    # print("    Applied uniform smoothing")
                elif smoothing_method == 'median':
                    Z_smooth = median_filter(Z, size=int(2 * sigma + 1))
                    # print("    Applied median smoothing")
                elif smoothing_method == 'savgol':
                    Z_smooth = smooth_savgol(Z, sigma)
                elif smoothing_method == 'bilateral':
                    Z_smooth = smooth_bilateral(Z, sigma)
                elif smoothing_method == 'anisotropic':
                    Z_smooth = denoise_tv_chambolle(Z, weight=sigma)
                elif smoothing_method == 'rebuild':
                    X, Y, Z_smooth = rebuild_and_resample_surface(X, Y, Z, grid_shape)
                else:
                    raise ValueError(f"Unsupported smoothing method: {smoothing_method}")

                # Create intrinsic UV coordinates based on regular grid shape
                nx, ny = grid_shape
                U, V = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

                # Compute derivatives with respect to intrinsic surface coordinates U and V
                du = dv = 1.0
                Zu, Zv = np.gradient(Z_smooth, du, dv)
                Zuu, Zuv = np.gradient(Zu, du, dv)
                _, Zvv = np.gradient(Zv, du, dv)

                # First fundamental form (intrinsic to surface grid)
                E = Zu ** 2 + 1
                F = Zu * Zv
                G = Zv ** 2 + 1

                # Second fundamental form
                L = Zuu
                M = Zuv
                N = Zvv

                # Compute principal curvatures using intrinsic UV derivatives
                denom = E * G - F ** 2
                sqrt_term = np.sqrt(
                    np.clip((L * G - 2 * M * F + N * E) ** 2 - 4 * denom * (L * N - M ** 2), a_min=0, a_max=None))
                k1 = ((L * G - 2 * M * F + N * E) + sqrt_term) / (2 * denom)
                k2 = ((L * G - 2 * M * F + N * E) - sqrt_term) / (2 * denom)

                # Flatten and reorder based on absolute magnitude
                k1_flat = k1.flatten(order='F')
                k2_flat = k2.flatten(order='F')
                abs_k1 = np.abs(k1_flat)
                abs_k2 = np.abs(k2_flat)

                use_k1_as_max = abs_k1 >= abs_k2

                max_curv = np.where(use_k1_as_max, k1_flat, k2_flat)
                min_curv = np.where(use_k1_as_max, k2_flat, k1_flat)

                # Compute gradients using global XYZ coordinates for direction calculations
                Zx, Zy = np.gradient(Z_smooth, dx, dy)
                Zxx, Zxy = np.gradient(Zx, dx, dy)
                _, Zyy = np.gradient(Zy, dx, dy)

                # Compute principal curvature directions using Hessian eigen-decomposition
                principal_dirs_max = np.zeros((nx, ny, 3))
                principal_dirs_min = np.zeros((nx, ny, 3))

                for ix in range(nx):
                    for iy in range(ny):
                        # Construct Hessian of the height function
                        H = np.array([[Zxx[ix, iy], Zxy[ix, iy]],
                                      [Zxy[ix, iy], Zyy[ix, iy]]])

                        # Eigen decomposition (curvature directions in XY plane)
                        eigvals, eigvecs = np.linalg.eigh(H)

                        # Sort by absolute curvature
                        idx = np.argsort(np.abs(eigvals))[::-1]  # descending
                        v_max = eigvecs[:, idx[0]]  # eigenvector of max abs curvature
                        v_min = eigvecs[:, idx[1]]  # eigenvector of min abs curvature

                        # Convert to 3D direction vectors
                        dzdx = Zx[ix, iy]
                        dzdy = Zy[ix, iy]
                        normal = np.array([-dzdx, -dzdy, 1.0])
                        normal /= np.linalg.norm(normal)

                        # Extend eigenvectors into 3D tangent directions
                        d_max = np.array([v_max[0], v_max[1], v_max[0] * dzdx + v_max[1] * dzdy])
                        d_min = np.array([v_min[0], v_min[1], v_min[0] * dzdx + v_min[1] * dzdy])

                        d_max -= np.dot(d_max, normal) * normal  # project to tangent plane
                        d_min -= np.dot(d_min, normal) * normal

                        d_max /= np.linalg.norm(d_max)
                        d_min /= np.linalg.norm(d_min)

                        # Logging for inspection
                        # print(f"Point ({ix}, {iy})")
                        # print(f"  Location: X={X[ix, iy]}, Y={Y[ix, iy]}, Z={Z[ix, iy]}")
                        # print(f"  Gradients: Zx={dzdx:.6f}, Zy={dzdy:.6f}")
                        # print(f"  Normal: {normal}")
                        # print(f"  Hessian Eigenvalues: {eigvals}")
                        # print(f"  Max dir (3D): {d_max}")
                        # print(f"  Min dir (3D): {d_min}")

                        principal_dirs_max[ix, iy] = d_max
                        principal_dirs_min[ix, iy] = d_min

                # Scale by curvature magnitude (cm)
                max_vecs = (principal_dirs_max * np.abs(max_curv).reshape((nx, ny), order='F')[:, :,
                                                 np.newaxis]).reshape(-1, 3, order='F')
                min_vecs = (principal_dirs_min * np.abs(min_curv).reshape((nx, ny), order='F')[:, :,
                                                 np.newaxis]).reshape(-1, 3, order='F')

                # Overwrite DataFrame columns
                df['Max Curvature Length'] = max_curv
                df['Min Curvature Length'] = min_curv
                df['Max Curvature Direction'] = ["{" + f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}" + "}" for v in max_vecs]
                df['Min Curvature Direction'] = ["{" + f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}" + "}" for v in min_vecs]

                df.to_excel(writer, sheet_name=sheet_name, index=False)
                # print("    Sheet saved with updated curvature")

                # Save debug plots for first 3 sheets
                if i < 5:
                    fig = plt.figure(figsize=(48, 14))

                    # Left: original surface colored by deformation magnitude
                    ax0 = fig.add_subplot(131, projection='3d')
                    deformation = np.abs(Z - Z_smooth)
                    max_def = np.nanmax(deformation)
                    deformation_norm = deformation / max_def if max_def != 0 else np.zeros_like(deformation)
                    surf0 = ax0.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(deformation_norm), linewidth=0,
                                             antialiased=False)
                    ax0.set_title('Original Surface - Deformation Magnitude', pad=20)
                    mappable = plt.cm.ScalarMappable(cmap='viridis')
                    mappable.set_array(deformation)
                    fig.colorbar(mappable, ax=ax0, shrink=0.5, pad=0.2, aspect=10, label='Deformation (cm)')

                    # Center: max curvature vectors
                    ax1 = fig.add_subplot(132, projection='3d')
                    ax1.plot_surface(X, Y, Z_smooth, cmap='Blues', alpha=0.5)
                    ax1.quiver(X, Y, Z_smooth, max_vecs[:, 0].reshape((nx, ny), order='F'),
                               max_vecs[:, 1].reshape((nx, ny), order='F'), max_vecs[:, 2].reshape((nx, ny), order='F'),
                               color='black', length=8, normalize=False, linewidths=0.5)
                    ax1.set_title('Max Curvature Vectors', pad=40)

                    # Right: min curvature vectors
                    ax2 = fig.add_subplot(133, projection='3d')
                    ax2.plot_surface(X, Y, Z_smooth, cmap='Greens', alpha=0.5)
                    ax2.quiver(X, Y, Z_smooth, min_vecs[:, 0].reshape((nx, ny), order='F'),
                               min_vecs[:, 1].reshape((nx, ny), order='F'), min_vecs[:, 2].reshape((nx, ny), order='F'),
                               color='black', length=8, normalize=False, linewidths=0.5)
                    ax2.set_title('Min Curvature Vectors', pad=40)

                    plt.tight_layout()
                    plt.savefig(debug_output_dir / f"{file_path.stem}_sheet{i}_curvature_vectors.png")
                    plt.close()
                    print(f"    Debug vector plot saved: sheet {i}")

            except Exception as e:
                print(f"  Error processing {sheet_name}: {e}")

                # --- DEBUGGING BLOCK ---
                try:
                    print("    DEBUG: Checking for NaNs or Infs")
                    print("      Z contains NaN?", np.isnan(Z).any())
                    print("      Z contains Inf?", np.isinf(Z).any())
                    print("      Zx range:", np.nanmin(Zx), "to", np.nanmax(Zx))
                    print("      Zy range:", np.nanmin(Zy), "to", np.nanmax(Zy))
                    print("      k1 range:", np.nanmin(k1), "to", np.nanmax(k1))
                    print("      k2 range:", np.nanmin(k2), "to", np.nanmax(k2))
                except Exception as debug_e:
                    print("    DEBUG: Could not compute debug info:", debug_e)
                import traceback
                traceback.print_exc()
                # --- END DEBUGGING BLOCK ---

                df.to_excel(writer, sheet_name=sheet_name, index=False)

        writer.close()
        print(f"Saved: {output_path}")
