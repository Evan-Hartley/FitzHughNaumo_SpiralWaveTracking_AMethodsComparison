import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

# Change variables to match Simulation and Output parameters
grid_size = (100, 100)
results_folder_name = "../Results/Test/"
output_name = "u_vals_evolution"
plot_title = "Voltage Evolution"


# Find and sort all u_vals files
file_list = sorted(glob.glob(results_folder_name + "u_vals_t*.bin"), key=lambda x: int(''.join(filter(str.isdigit, x))))

# Load data
frames = []
for filename in file_list:
    data = np.fromfile(filename, dtype=np.double).reshape(grid_size)

    if data.shape == grid_size:
        frames.append(data)
    else:
        print(f"Skipping {filename}: shape {data.shape} != {grid_size}")

# Create animation
fig, ax = plt.subplots()
cax = ax.imshow(frames[0], cmap='rainbow', interpolation='nearest', vmin=-0.6, vmax=1.4)
fig.colorbar(cax)
ax.set_title(plot_title)

# Fill animation
def update(frame):
    cax.set_array(frame)
    return [cax]

# Compile animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, blit=True)

# Save animation
ani.save(results_folder_name + output_name + ".gif", writer='pillow')
print("Saved gif")
plt.close()
