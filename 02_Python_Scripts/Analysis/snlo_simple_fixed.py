import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt

print("="*60)
print("BBO CRYSTAL DATA COLLECTION")
print("="*60)

data = {
    'wavelength_nm': [400, 500, 583, 600, 700, 800, 900, 1000],
    'n_o': [1.690, 1.680, 1.670, 1.670, 1.665, 1.655, 1.650, 1.645],
    'n_e': [1.565, 1.550, 1.552, 1.545, 1.540, 1.535, 1.530, 1.525]
}

print("\nData Summary:")
print("-"*50)
print(f"{'λ (nm)':<10} {'n_o':<10} {'n_e':<10} {'Δn':<10}")
print("-"*50)

for i in range(len(data['wavelength_nm'])):
    delta = data['n_o'][i] - data['n_e'][i]
    print(f"{data['wavelength_nm'][i]:<10} {data['n_o'][i]:<10.3f} {data['n_e'][i]:<10.3f} {delta:<10.3f}")

print("-"*50)

delta_n = [data['n_o'][i] - data['n_e'][i] for i in range(len(data['wavelength_nm']))]

df = pd.DataFrame({
    'Wavelength_nm': data['wavelength_nm'],
    'n_o': data['n_o'],
    'n_e': data['n_e'],
    'birefringence_delta_n': delta_n
})
df.to_csv('BBO_data.csv', index=False)
print("\n✓ Data saved to 'BBO_data.csv'")

plt.figure(figsize=(12, 8))
plt.plot(data['wavelength_nm'], data['n_o'], 'bo-', linewidth=2.5, markersize=8, label='n_o (ordinary)')
plt.plot(data['wavelength_nm'], data['n_e'], 'rs-', linewidth=2.5, markersize=8, label='n_e (extraordinary)')
plt.plot(data['wavelength_nm'], delta_n, 'g^--', linewidth=2, markersize=8, label='Delta n (birefringence)')

plt.xlabel('Wavelength (nm)', fontsize=14)
plt.ylabel('Refractive Index', fontsize=14)
plt.title('BBO Crystal: Refractive Indices vs Wavelength', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

plt.axvline(x=583, color='purple', linestyle=':', alpha=0.8, linewidth=2)
plt.text(585, 1.53, 'Signal (583 nm)', fontsize=11, color='purple')
plt.axvline(x=900, color='orange', linestyle=':', alpha=0.8, linewidth=2)
plt.text(905, 1.53, 'Idler (900 nm)', fontsize=11, color='orange')

plt.savefig('BBO_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('BBO_plot.pdf', bbox_inches='tight')
plt.close()

print("\n✓ Plot saved to 'BBO_plot.png' and 'BBO_plot.pdf'")

print("\n" + "="*60)
print("COMPLETE!")
print("Files created:")
print("  - BBO_data.csv")
print("  - BBO_plot.png")
print("  - BBO_plot.pdf")
print("="*60)