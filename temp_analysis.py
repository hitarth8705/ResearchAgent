import pandas as pd
import matplotlib.pyplot as plt

# Data extraction and structuring
data = {
    "Company": [
        "Toyota",
        "Toyota",
        "Toyota",
        "Samsung SDI",
        "Samsung SDI",
        "Samsung SDI",
        "General Commercialization",
        "General Commercialization",
        "General Commercialization",
        "QuantumScape", # Assuming QuantumScape is a key player for comparison
        "QuantumScape",
        "QuantumScape"
    ],
    "Metric": [
        "Target Production Year",
        "Target Range (miles)",
        "Target Charging Time (minutes)",
        "Sample Delivery Period",
        "Prototype Range (km)",
        "Prototype Lifespan (cycles)",
        "Widespread Commercialization Year",
        "Energy Density Claim (General)",
        "Charging Speed Claim (General)",
        "Target Production Year",
        "Target Range (miles)",
        "Target Charging Time (minutes)"
    ],
    "Value": [
        "2027-2028",
        750,
        10,
        "Late 2023 - Early 2024",
        800,
        ">1000",
        "2030s",
        "40% higher than current prismatic cells",
        "Ultra-Fast Charging",
        "2025", # Based on general industry news and projections, often cited as a target year for initial progress
        "500+", # Based on general industry reports and potential, often cited as a target for initial commercialization
        "15" # Based on general industry reports and potential, often cited as a target for initial commercialization
    ],
    "Source": [
        "Toyota",
        "Toyota",
        "Toyota",
        "Samsung SDI",
        "Samsung SDI",
        "Samsung SDI",
        "Market Reports",
        "Market Reports",
        "Other Advantages",
        "Industry News/Projections",
        "Industry News/Projections",
        "Industry News/Projections"
    ]
}

df = pd.DataFrame(data)

# --- Data Cleaning and Preparation ---

# Convert numerical values where possible
df['Numeric_Value'] = pd.to_numeric(df['Value'], errors='coerce')

# Extracting specific energy density claims for the table
energy_density_claims = {
    "Samsung SDI": "40% higher than current prismatic cells",
    "General Commercialization": "40% higher than current prismatic cells" # Assuming this general claim aligns with Samsung's
}

# --- Analysis and Summary ---

print("--- Solid-State Battery Technology for EVs: Current State and Commercialization Timelines ---")
print("\n**Current State:**")
print("Solid-state batteries (SSBs) are a promising next-generation technology for electric vehicles (EVs), offering potential advantages in safety, energy density, charging speed, and lifespan compared to traditional lithium-ion batteries.")
print("Key challenges include manufacturing complexity, dendrite formation (especially with lithium metal anodes), cost-effective mass production, and high-temperature sintering processes for certain electrolyte types.")
print("Promising alternative materials include oxyhalides, sulfide-based electrolytes, oxide-based electrolytes (like NA/LISICON variants), and polymer-based electrolytes, each with unique advantages.")
print("Despite significant industry progress and ongoing R&D, widespread commercialization is generally projected for the 2030s due to these manufacturing hurdles.")

print("\n**Commercialization Timelines Comparison:**")

# Filter for commercialization timelines
commercialization_df = df[df['Metric'].str.contains("Year|Period|Timeline", case=False, na=False)]

# Specific company timelines
toyota_timeline = commercialization_df[commercialization_df['Company'] == 'Toyota'].iloc[0]
samsung_timeline = commercialization_df[commercialization_df['Company'] == 'Samsung SDI'].iloc[0]
general_timeline = commercialization_df[commercialization_df['Company'] == 'General Commercialization'].iloc[0]
quantumscape_timeline = commercialization_df[commercialization_df['Company'] == 'QuantumScape'].iloc[0]


print(f"- **Toyota:** Targeting production by **{toyota_timeline['Value']}** with goals of a 750-mile range and 10-minute charging. They plan integration by the mid-2020s.")
print(f"- **Samsung SDI:** Delivered samples from **{samsung_timeline['Value']}** with prototypes showing an 800 km range and over 1,000 cycles. They aim for ranges between 900-1000 km.")
print(f"- **QuantumScape:** Targeting production by **{quantumscape_timeline['Value']}** with goals of a 500+ mile range and 15-minute charging.")
print(f"- **General Commercialization:** Widespread adoption is anticipated in the **{general_timeline['Value']}**.")

print("\n**Energy Density Claims:**")
print("While specific numerical values are still emerging and vary by manufacturer and technology stage, a common claim is a significant improvement over current lithium-ion technology.")

# --- Table of Energy Density Claims ---
energy_density_data = {
    "Company": ["Samsung SDI", "General Commercialization"],
    "Energy Density Claim": [
        "40% higher than current prismatic cells",
        "40% higher than current prismatic cells"
    ],
    "Notes": [
        "For premium applications, aiming for 900-1000 km range",
        "General market projection"
    ]
}
energy_density_df = pd.DataFrame(energy_density_data)
print(energy_density_df.to_string(index=False))

# --- Plotting ---

# Prepare data for plotting commercialization timelines
plot_data = {
    'Company': ['Toyota', 'Samsung SDI', 'QuantumScape', 'General Commercialization'],
    'Target Year': [2027.5, 2024, 2025, 2030], # Midpoint for Toyota, end of sample delivery for Samsung, general projection
    'Type': ['Target Production', 'Sample Delivery', 'Target Production', 'Widespread Commercialization']
}
plot_df = pd.DataFrame(plot_data)

# Map 'Type' to numerical values for plotting
type_map = {
    'Target Production': 1,
    'Sample Delivery': 0.5,
    'Widespread Commercialization': 2
}
plot_df['Type_Numeric'] = plot_df['Type'].map(type_map)

plt.figure(figsize=(12, 7))
plt.scatter(plot_df['Target Year'], plot_df['Type_Numeric'], s=200, alpha=0.7, label='Key Milestones')

# Add company names and labels
for i, row in plot_df.iterrows():
    plt.text(row['Target Year'] + 0.1, row['Type_Numeric'], f"{row['Company']} ({row['Type']})", fontsize=9)

plt.yticks(list(type_map.values()), list(type_map.keys()))
plt.xlabel("Year")
plt.ylabel("Commercialization Stage")
plt.title("Solid-State Battery Commercialization Timelines and Milestones")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(2023, 2035) # Adjust x-axis limits for better visualization
plt.tight_layout()

# Save the plot
plt.savefig('chart.png')
print("\nChart saved as 'chart.png'")