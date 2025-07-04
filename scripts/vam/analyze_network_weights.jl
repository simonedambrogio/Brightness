using Pkg
Pkg.activate("/Volumes/PROJECTS/Ongoing/Salience/scr/VAMSalience")

using NPZ, YAML, LinearAlgebra, CairoMakie, GLMakie, Statistics, ProgressBars, MultivariateStats

println("--- Setting up Configuration ---")
config = YAML.load_file("config.yaml");

function load_activations(batch_idx, checkpoint, activation_layer_name)
    # Construct filenames
    batch_str = lpad(string(batch_idx), 4, '0');
    # Ensure checkpoint string matches Python output ("ckpt" prefix)
    ckpt_str = "ckpt$(checkpoint)";
    activation_filename = "activations_$(activation_layer_name)_$(ckpt_str)_batch$(batch_str).npy";
    activation_path = joinpath(activation_base_dir, activation_filename);
    activations = npzread(activation_path);
    return activations
end;

batch_idx = 0;       # Load the first batch (index 0)
activation_layer_name = "Dense_1_input";
num_behavioral_vars = 4;
num_activation_features = 1024; # Expected number of features from Dense_1 input

# ====== Brightness ======
activation_base_dir = joinpath(
    config["local"]["data"], "results", "vam", "weights"
);
input_base_dir = joinpath(config["local"]["data"], "processed", "vam");
checkpoint = 69;
n_trials = 35000;
batch_size = 256;
batch_max = n_trials ÷ batch_size;

# Load activations for all batches --------------------------------------------
println("Loading activations")
activations = vcat([
    load_activations(batch_idx, checkpoint, activation_layer_name)
    for batch_idx in ProgressBar(0:(batch_max-1))
]...);
n_trials = size(activations, 1);    

# After loading activations
sparsity_percentage = 100 * sum(activations .== 0) / length(activations);
println("Sparsity: $sparsity_percentage%")

# Load behavioral data --------------------------------------------------------
rts_path = joinpath(input_base_dir, "rts.npy");
gain_path = joinpath(input_base_dir, "value_gain.npy");
loss_path = joinpath(input_base_dir, "value_loss.npy");
left_loss_path = joinpath(input_base_dir, "left_loss.npy");
salience_path = joinpath(input_base_dir, "salience.npy");
files_to_check = [gain_path, loss_path, left_loss_path, salience_path];
[error("Required data file not found: $f") for f in files_to_check if !isfile(f)];

rts = npzread(rts_path)[1:n_trials];
value_gain = npzread(gain_path)[1:n_trials];
value_loss = npzread(loss_path)[1:n_trials];
left_loss = npzread(left_loss_path)[1:n_trials];
# Rename variable to avoid confusion: 0=LossSal, 1=GainSal
gain_salient = npzread(salience_path)[1:n_trials]; 
value_left = [Bool(left_loss) ? gain : loss for (gain, loss, left_loss) in zip(value_gain, value_loss, left_loss)];
value_right = [Bool(left_loss) ? loss : gain for (gain, loss, left_loss) in zip(value_gain, value_loss, left_loss)];
left_salient = @. (Bool(gain_salient) & !Bool(left_loss)) | (!Bool(gain_salient) & Bool(left_loss)) # Use renamed var
println("Sliced Shapes: Gain=$(size(value_gain)), Loss=$(size(value_loss)), LeftLoss=$(size(left_loss)), SalienceNumeric=$(size(gain_salient))")



println("\n--- Preprocessing Activations for PCA ---")
feature_variances = Statistics.var(activations, dims=1); # Result is 1xN matrix
tolerance = 1e-8;
non_zero_var_indices = findall(>(tolerance), vec(feature_variances)); # Get linear indices
num_original_features = size(activations, 2);
num_active_features = length(non_zero_var_indices);
println("Original features: $num_original_features")
println("Features with non-zero variance (> $tolerance): $num_active_features")
if num_active_features == 0
    error("No features with non-zero variance found. Cannot perform PCA.")
end
activations_filtered = activations[:, non_zero_var_indices];
println("Filtered activations shape: ", size(activations_filtered));



println("\n--- Performing PCA using MultivariateStats.jl ---")
# Prepare data: MultivariateStats expects features in rows, observations in columns
# We also typically center the data for PCA (though fit handles it with center=true)
data_for_pca = activations_filtered';  # Transpose: Now Features x Trials
println("Shape of data for PCA (Features x Trials): ", size(data_for_pca))
# Fit the PCA model
# pratio = 1.0 ensures we capture all variance from the active features
# maxoutdim can be used to limit components, e.g., maxoutdim=10
# Remove center=true; mean=nothing (default) handles centering
M = fit(PCA, data_for_pca; pratio=1.0);
println("PCA Model Fitted.")

# Analyze PCA results
explained_variance_ratio = principalvars(M) ./ tvar(M);
println("Explained variance ratio per component:")
[println("Comp $i: $(explained_variance_ratio[i] * 100)%") for i in 1:3];
println("\nTotal variance explained by the first 3 components: ", sum(explained_variance_ratio[1:3]) * 100);

# Project data onto principal components
# The result (Y) will have components in rows, observations in columns
projected_data = predict(M, data_for_pca);
println("\nShape of projected data (Components x Trials): ", size(projected_data));

# If you want it back in Trials x Components format:
projected_data_trials_x_comps = projected_data';
println("Shape of projected data (Trials x Components): ", size(projected_data_trials_x_comps));

comp1, comp2, comp3 = projected_data_trials_x_comps[:, 1], projected_data_trials_x_comps[:, 2], projected_data_trials_x_comps[:, 3];

valdiff_index = [value_gain .- value_loss .== i for i in -6:6];
left_index = Bool.(left_loss);
right_index = .!left_index;
gain_sal_index = Bool.(gain_salient);
loss_sal_index = .!gain_sal_index;
valdiff_comp1 = [mean(comp1[i]) for i in valdiff_index];
begin
    fig = Figure(size = (550, 550), fontsize = 28, fonts = (; regular = "Avenir Light"));
    ax = Axis(
        fig[1, 1], xlabel = "Value Difference\n(Gain - Loss)", 
        ylabel = "Principal Component 1",
        xgridvisible = false, ygridvisible = false,
        xtrimspine = true, ytrimspine = true,
        topspinevisible = false, rightspinevisible = false,
        limits = (-7, 7, -21, 21),
        xticks = -6:3:6,
    );
    # Calculate means using CORRECTED indices
    # Mean PC1 for trials where LOSS was salient
    valdiff_losssal_comp1 = [mean(comp1[i .& loss_sal_index]) for i in valdiff_index] 
    # Mean PC1 for trials where GAIN was salient
    valdiff_gainsal_comp1 = [mean(comp1[i .& gain_sal_index]) for i in valdiff_index]
    
    vlines!(ax, 0, color = :black, linestyle = (:dash, :loose), linewidth = 0.8)
    hlines!(ax, 0, color = :black, linestyle = (:dash, :loose), linewidth = 0.8)

    # Plot with CORRECT labels and markers
    # Gain Salient (calculated using gain_salient_index)
    scatter!(ax, -6:6, valdiff_gainsal_comp1, color = config["colors"]["gain-salient"], colormap = :viridis, 
             markersize = 16, label="Gain Salient")
    # Loss Salient (calculated using loss_salient_index)
    scatter!(ax, -6:6, valdiff_losssal_comp1, color = config["colors"]["loss-salient"], colormap = :viridis, 
             markersize = 16, label="Loss Salient")

    axislegend(ax, position = :lt, framevisible = false, labelsize=22) # Legend position might need adjustment
    # Optional: Add a colorbar if helpful, though color matches x-axis here
    # Colorbar(fig[1, 2], limits=(-6, 6), colormap=:viridis, label="Value Difference")
    # display(fig)
end

# save the figure
CairoMakie.save("figures/vam/pca_dense1_valdiff.svg", fig)
CairoMakie.save("figures/vam/pca_dense1_valdiff.png", fig, px_per_unit = 4)

println("cor(value_left, comp1) = ", Statistics.cor(value_left, comp1))
println("cor(value_left, comp2) = ", Statistics.cor(value_left, comp2))
println("cor(value_left, comp3) = ", Statistics.cor(value_left, comp3))

println("cor(value_right, comp1) = ", Statistics.cor(value_right, comp1))
println("cor(value_right, comp2) = ", Statistics.cor(value_right, comp2))
println("cor(value_right, comp3) = ", Statistics.cor(value_right, comp3))

println("cor(value_left .- value_right, comp1) = ", Statistics.cor(value_left .- value_right, comp1))
println("cor(value_left .- value_right, comp2) = ", Statistics.cor(value_left .- value_right, comp2))
println("cor(value_left .- value_right, comp3) = ", Statistics.cor(value_left .- value_right, comp3))

println("cor(value_gain, comp1) = ", Statistics.cor(value_gain, comp1))
println("cor(value_gain, comp2) = ", Statistics.cor(value_gain, comp2))
println("cor(value_gain, comp3) = ", Statistics.cor(value_gain, comp3))

println("cor(value_loss, comp1) = ", Statistics.cor(value_loss, comp1))
println("cor(value_loss, comp2) = ", Statistics.cor(value_loss, comp2))
println("cor(value_loss, comp3) = ", Statistics.cor(value_loss, comp3))

println("cor(value_gain .- value_loss, comp1) = ", Statistics.cor(value_gain .- value_loss, comp1))
println("cor(value_gain .- value_loss, comp2) = ", Statistics.cor(value_gain .- value_loss, comp2))
println("cor(value_gain .- value_loss, comp3) = ", Statistics.cor(value_gain .- value_loss, comp3))

println("cor(gain_salient, comp1) = ", Statistics.cor(gain_salient, comp1))
println("cor(gain_salient, comp2) = ", Statistics.cor(gain_salient, comp2))
println("cor(gain_salient, comp3) = ", Statistics.cor(gain_salient, comp3))

println("cor(left_salient, comp1) = ", Statistics.cor(left_salient, comp1))
println("cor(left_salient, comp2) = ", Statistics.cor(left_salient, comp2))
println("cor(left_salient, comp3) = ", Statistics.cor(left_salient, comp3))
# ========================


