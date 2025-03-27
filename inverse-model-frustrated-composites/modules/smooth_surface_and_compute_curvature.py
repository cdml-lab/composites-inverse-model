# ┌───────────────────────────────────────────────────────────────────────────┐
# │                        Surface Smoothing & Curvature                     │
# └───────────────────────────────────────────────────────────────────────────┘

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def smooth_surface_and_compute_curvature(base_dir, input_files_list, grid_shape, smoothing_method='gaussian', sigma=1.0):
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
        print(f"Processing: {file_path.name}")
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        output_path = file_path.with_name(file_path.stem + "_smooth_curvature.xlsx")
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

        debug_output_dir = file_path.parent / "debug_plots"
        debug_output_dir.mkdir(exist_ok=True)

        for i, sheet_name in enumerate(sheet_names):
            print(f"  Sheet: {sheet_name}")
            df = pd.read_excel(xls, sheet_name=sheet_name)

            required_cols = ['Location X', 'Location Y', 'Location Z']
            if not all(col in df.columns for col in required_cols):
                print(f"  Skipping {sheet_name} (missing XYZ columns)")
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                continue

            try:
                # Validate size
                expected_size = grid_shape[0] * grid_shape[1]
                if len(df) != expected_size:
                    raise ValueError(f"Expected {expected_size} points for shape {grid_shape}, got {len(df)}")

                # Reshape into 2D grid
                nx, ny = grid_shape
                dx = dy = 1.0  # cm spacing
                X = df['Location X'].to_numpy().reshape((nx, ny), order='F')
                Y = df['Location Y'].to_numpy().reshape((nx, ny), order='F')
                Z = df['Location Z'].to_numpy().reshape((nx, ny), order='F')

                print("    Reshaped XYZ into grid")

                # Smooth Z using selected method
                if smoothing_method == 'gaussian':
                    from scipy.ndimage import gaussian_filter
                    Z_smooth = gaussian_filter(Z, sigma=sigma)
                    print("    Applied Gaussian smoothing")
                elif smoothing_method == 'uniform':
                    from scipy.ndimage import uniform_filter
                    Z_smooth = uniform_filter(Z, size=int(2*sigma+1))
                    print("    Applied uniform smoothing")
                elif smoothing_method == 'median':
                    from scipy.ndimage import median_filter
                    Z_smooth = median_filter(Z, size=int(2*sigma+1))
                    print("    Applied median smoothing")
                else:
                    raise ValueError(f"Unsupported smoothing method: {smoothing_method}")

                # Compute gradients using spacing-aware finite differences
                Zx, Zy = np.gradient(Z_smooth, dx, dy)
                Zxx, Zxy = np.gradient(Zx, dx, dy)
                _, Zyy = np.gradient(Zy, dx, dy)

                print("    Computed derivatives")

                # First fundamental form coefficients
                E = 1 + Zx**2
                F = Zx * Zy
                G = 1 + Zy**2

                # Second fundamental form coefficients
                L = Zxx
                M = Zxy
                N = Zyy

                # Compute principal curvatures
                denom = E * G - F**2
                sqrt_term = np.sqrt(np.clip((L * G - 2 * M * F + N * E)**2 - 4 * denom * (L * N - M**2), a_min=0, a_max=None))
                k1 = ((L * G - 2 * M * F + N * E) + sqrt_term) / (2 * denom)
                k2 = ((L * G - 2 * M * F + N * E) - sqrt_term) / (2 * denom)

                print("    Calculated principal curvatures")

                # Flatten and reorder based on absolute magnitude
                k1_flat = k1.flatten(order='F')
                k2_flat = k2.flatten(order='F')
                abs_k1 = np.abs(k1_flat)
                abs_k2 = np.abs(k2_flat)

                use_k1_as_max = abs_k1 >= abs_k2

                max_curv = np.where(use_k1_as_max, k1_flat, k2_flat)
                min_curv = np.where(use_k1_as_max, k2_flat, k1_flat)

                # Estimate direction vectors (unit vectors in X/Y/Z)
                normals = np.dstack((-Zx, -Zy, np.ones_like(Zx)))
                normals /= np.linalg.norm(normals, axis=2, keepdims=True)

                # For simplicity, we'll set both directions perpendicular to normal and X or Y
                dirs_max = np.dstack((np.ones_like(Zx), np.zeros_like(Zx), Zx))
                dirs_min = np.dstack((np.zeros_like(Zx), np.ones_like(Zx), Zy))

                dirs_max /= np.linalg.norm(dirs_max, axis=2, keepdims=True)
                dirs_min /= np.linalg.norm(dirs_min, axis=2, keepdims=True)

                # Scale by curvature magnitude (cm)
                max_vecs = (dirs_max * np.abs(max_curv).reshape((nx, ny), order='F')[:, :, np.newaxis]).reshape(-1, 3, order='F')
                min_vecs = (dirs_min * np.abs(min_curv).reshape((nx, ny), order='F')[:, :, np.newaxis]).reshape(-1, 3, order='F')

                # Overwrite DataFrame columns
                df['Max Curvature Length'] = max_curv
                df['Min Curvature Length'] = min_curv
                df['Max Curvature Direction'] = ["{" + f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}" + "}" for v in max_vecs]
                df['Min Curvature Direction'] = ["{" + f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}" + "}" for v in min_vecs]

                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print("    Sheet saved with updated curvature")

                # Save debug plots for first 3 sheets
                if i < 5:
                    fig = plt.figure(figsize=(28, 14))

                    ax1 = fig.add_subplot(121, projection='3d')
                    ax1.plot_surface(X, Y, Z_smooth, cmap='Blues', alpha=0.5)
                    ax1.quiver(X, Y, Z_smooth, max_vecs[:, 0].reshape((nx, ny), order='F'), max_vecs[:, 1].reshape((nx, ny), order='F'), max_vecs[:, 2].reshape((nx, ny), order='F'), color='black', length=8, normalize=False, linewidths=0.5)
                    ax1.set_title('Max Curvature Vectors')

                    ax2 = fig.add_subplot(122, projection='3d')
                    ax2.plot_surface(X, Y, Z_smooth, cmap='Greens', alpha=0.5)
                    ax2.quiver(X, Y, Z_smooth, min_vecs[:, 0].reshape((nx, ny), order='F'), min_vecs[:, 1].reshape((nx, ny), order='F'), min_vecs[:, 2].reshape((nx, ny), order='F'), color='black', length=8, normalize=False,linewidths=0.5)
                    ax2.set_title('Min Curvature Vectors')

                    plt.tight_layout()
                    plt.savefig(debug_output_dir / f"{file_path.stem}_sheet{i}_curvature_vectors.png")
                    plt.close()
                    print(f"    Debug vector plot saved: sheet {i}")

            except Exception as e:
                print(f"  Error processing {sheet_name}: {e}")
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        writer.close()
        print(f"Saved: {output_path}")
