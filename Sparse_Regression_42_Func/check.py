import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
#from main import *
import torch
from model import *
from data import input, output, df

model = CustomModel(42)
model.load_state_dict(torch.load('model_weights.pth'))

for key, value in model.state_dict().items():
    if 'weight' in key:
        print(key)
        print(value)

    if 'bias' in key:
        print(key)
        print(value)


# Create symbolic variables for P, T, and rho
P, T, rho = sp.symbols('P T rho')

# # Calculate the symbolic expression for rho
rho_expression = symbolic_expression(P, T, model.state_dict())

print()
print("Printing the Symbolic Expression for rho_r:")
print(rho_expression)
print()
print(sp.simplify(sp.expand(rho_expression[0])))
# print(sp.simplify(sp.expand(rho_expression)))

combined_dataset = TensorDataset(input, output)
check_dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=False)

pred_outputs = []
model.eval()
with torch.no_grad():

    for i, data in enumerate(check_dataloader, 0):
        inputs, labels = data

        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        predictions = model(inputs)
        pred_outputs.append(predictions)

pred_outputs = torch.cat(pred_outputs, dim=0)
ground_truth = output.numpy()
predictions = pred_outputs.numpy()

print("predictions = ", predictions)

plt.figure(figsize=(10, 5))
spacing = 200

t_index = [0, 49, 99, 149, 199]

for t in t_index:
    indices_to_plot = range(t, len(ground_truth), spacing)
    print()
    print("Plotting for Temperature = ", int(df.iloc[t][1]), " K")

    # Create a new figure
    #plt.figure()

    # Isotherms
    plt.plot(input[indices_to_plot, 0]*74, [ground_truth[i] for i in indices_to_plot], label="Ground Truth", marker='o', linestyle='None')
    plt.plot(input[indices_to_plot, 0]*74, [predictions[i] for i in indices_to_plot], label="Model Predictions", marker='x', linestyle='None')

    plt.title(f"Graph for Temperature = {int(df.iloc[t][1])} K")
    plt.xlabel("Pressure (in bars)")
    plt.ylabel("rho_reduced")
    plt.legend()
    plt.grid(True)
    # Save the figure after showing it
    # plt.savefig(f"Temperature_{int(df.iloc[t][1])}K.png")
    # plt.show()



# Isobars
p_index = [p for p in range(0, len(input),5000)]
for p in p_index:
    print()
    print("Plotting for Pressure = ", df.iloc[p][0], " bar")
    plt.plot(input[p:p+199, 1]*304, ground_truth[p:p+199], label="Ground Truth", marker='o', linestyle='None')
    plt.plot(input[p:p+199, 1]*304, predictions[p:p+199], label="Model Predictions", marker='x', linestyle='None')
    plt.title(f"Graph for Pressure = {df.iloc[p][0]:.1f} bar")
    plt.xlabel("Temperature (in K)")
    plt.ylabel("rho_reduced")
    plt.legend()
    plt.grid(True)
    # plt.savefig(f"Pressure_{df.iloc[p][0]:.1f}_bar.png")
    # plt.show()

