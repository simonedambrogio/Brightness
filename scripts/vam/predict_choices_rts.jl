using Pkg
Pkg.activate("/Volumes/PROJECTS/Ongoing/Salience/scr/VAMSalience")

using NPZ, YAML, LinearAlgebra, GLMakie, CairoMakie, Statistics, 
ProgressBars, MultivariateStats, DataFrames, Colors

println("--- Setting up Configuration ---")
config = YAML.load_file("config.yaml");

fontsize = 30;


# belnd 
function blend_colors(foreground; background="#FFFFFF", alpha=0.3)
    fg_rgb = convert(RGB, parse(Colorant, foreground))
    bg_rgb = convert(RGB, parse(Colorant, background))
    
    # Use broadcasting for the blend calculation
    blended = alpha .* (fg_rgb.r, fg_rgb.g, fg_rgb.b) .+ (1 - alpha) .* (bg_rgb.r, bg_rgb.g, bg_rgb.b)
    
    return RGB(blended...)
end


function load_predictions(activation_base_dir, checkpoint)
    # Ensure checkpoint string matches Python output ("ckpt" prefix)
    # ckpt69_train_actual.npy
    train_actual = "ckpt$(checkpoint)_train_actual_full.npy";
    train_simulated = "ckpt$(checkpoint)_train_simulated_full.npy";
    val_actual = "ckpt$(checkpoint)_val_actual_full.npy";
    val_simulated = "ckpt$(checkpoint)_val_simulated_full.npy";
    
    train_actual_path = joinpath(activation_base_dir, train_actual);
    train_simulated_path = joinpath(activation_base_dir, train_simulated);
    val_actual_path = joinpath(activation_base_dir, val_actual);
    val_simulated_path = joinpath(activation_base_dir, val_simulated);

    train_actual = npzread(train_actual_path);
    train_actual_df = DataFrames.DataFrame(train_actual, [:rt, :response, :salience, :condition, :value_gain, :value_loss], makeunique=true)
    train_simulated = npzread(train_simulated_path);
    train_simulated_df = DataFrames.DataFrame(train_simulated, [:rt, :response, :salience, :condition, :value_gain, :value_loss], makeunique=true)
    val_actual = npzread(val_actual_path);
    val_actual_df = DataFrames.DataFrame(val_actual, [:rt, :response, :salience, :condition, :value_gain, :value_loss], makeunique=true)
    val_simulated = npzread(val_simulated_path);
    val_simulated_df = DataFrames.DataFrame(val_simulated, [:rt, :response, :salience, :condition, :value_gain, :value_loss], makeunique=true)
        
    return (; train_actual = train_actual_df, train_simulated = train_simulated_df, val_actual = val_actual_df, val_simulated = val_simulated_df)
end;

function plot_rts(actual, simulated; color_gainsalient, color_lossalient, color_actual, backend = "GLMakie")

    if backend == "GLMakie"
        GLMakie.activate!()
    elseif backend == "CairoMakie"
        CairoMakie.activate!()
    end

    fig = Figure(size = (1000, 450), fontsize = fontsize, fonts = (; regular = "Avenir Light"));
    
    bdw = 0.35;  
    # Actual Data =====================================================
    rt_actual = 0.6 .+ actual.rt;
    range_idx = rt_actual .< 5 .&& rt_actual .> -5;
    gain_is_salient = Bool.(actual.salience[range_idx]);
    loss_is_salient = .!Bool.(actual.salience[range_idx]);
    rt_actual_gain = rt_actual[range_idx][gain_is_salient];
    rt_actual_loss = rt_actual[range_idx][loss_is_salient];
    # Model Predictions ================================================
    rt_simulated = 0.6 .+ simulated.rt;
    range_idx = rt_simulated .< 5 .&& rt_simulated .> -5;
    gain_is_salient = Bool.(actual.salience[range_idx]);
    loss_is_salient = .!Bool.(actual.salience[range_idx]);
    rt_simulated_gain = rt_simulated[range_idx][gain_is_salient];
    rt_simulated_loss = rt_simulated[range_idx][loss_is_salient];


    # Gain is Salient
    ax = Axis(
        fig[1, 1], xlabel = "RT",
        limits = ((-0.5, 5.5), (-0.05, 0.7)), 
        yticks = 0:0.2:0.6,
        xgridvisible = false, ygridvisible = false,
        xtrimspine = true, ytrimspine = true,
        # remve top and right spines
        topspinevisible = false, rightspinevisible = false,
    );
    density!(ax, rt_actual_gain, bandwidth = bdw, color = color_actual)
    density!(ax, rt_simulated_gain, bandwidth = bdw, color=(:blue, 0.0), strokecolor=color_gainsalient, strokewidth=10)
    # Loss is Salient
    ax = Axis(
        fig[1, 2], xlabel = "RT",
        limits = ((-0.5, 5.5), (-0.05, 0.7)), 
        yticks = 0:0.2:0.6,
        xgridvisible = false, ygridvisible = false,
        xtrimspine = true, ytrimspine = true,
        # remve top and right spines
        topspinevisible = false, rightspinevisible = false,
    );
    density!(ax, rt_actual_loss, bandwidth = bdw, color = color_actual)
    density!(ax, rt_simulated_loss, bandwidth = bdw, color=(:blue, 0.0), strokecolor=color_lossalient, strokewidth=10)

    if string(Makie.current_backend()) == "CairoMakie"
        println("Current backend is CairoMakie, the plot will not be displayed.")
        return fig
    else
        display(fig)
        return fig
    end
end;

function sem_bernoulli(data)
    n = length(data)
    if n == 0
        return NaN
    end
    p = mean(data) # Calculate the proportion of 1s
    # Handle cases where p is exactly 0 or 1 to avoid sqrt of negative (due to precision)
    if p == 0.0 || p == 1.0
        return 0.0
    end
     # Use max(0, ...) for robustness against potential floating point inaccuracies
    variance_mle = max(0.0, p * (1.0 - p)) 
    return sqrt(variance_mle / n)
end;

function get_paccept_bysalience(actual, simulated)
    # Actual Validation Data ===============================================
    # Gain is Salient
    value_diff_actual = (actual.value_gain .- actual.value_loss)[actual.salience .== 1];
    response = actual.response[actual.salience .== 1];
    p_accept_actual_gainsalient = [ mean(response[vd .== value_diff_actual]) for vd in -6:6];
    sem_accept_actual_gainsalient = [ sem_bernoulli(response[vd .== value_diff_actual]) for vd in -6:6];
    # Loss is Salient
    value_diff_actual = (actual.value_gain .- actual.value_loss)[actual.salience .== 0];
    response = actual.response[actual.salience .== 0];
    p_accept_actual_lossalient = [ mean(response[vd .== value_diff_actual]) for vd in -6:6];
    sem_accept_actual_lossalient = [ sem_bernoulli(response[vd .== value_diff_actual]) for vd in -6:6];

    # Simulated Validation Data ===============================================
    # Gain is Salient
    value_diff_simulated = (simulated.value_gain .- simulated.value_loss)[simulated.salience .== 1];
    response = simulated.response[simulated.salience .== 1];
    p_accept_simulated_gainsalient = [ mean(response[vd .== value_diff_simulated]) for vd in -6:6];
    sem_accept_simulated_gainsalient = [ sem_bernoulli(response[vd .== value_diff_simulated]) for vd in -6:6];
    # Loss is Salient
    value_diff_simulated = (simulated.value_gain .- simulated.value_loss)[simulated.salience .== 0];
    response = simulated.response[simulated.salience .== 0];
    p_accept_simulated_lossalient = [ mean(response[vd .== value_diff_simulated]) for vd in -6:6];
    sem_accept_simulated_lossalient = [ sem_bernoulli(response[vd .== value_diff_simulated]) for vd in -6:6];

    return DataFrames.DataFrame(; p_accept_actual_gainsalient, sem_accept_actual_gainsalient, p_accept_actual_lossalient, sem_accept_actual_lossalient, p_accept_simulated_gainsalient, sem_accept_simulated_gainsalient, p_accept_simulated_lossalient, sem_accept_simulated_lossalient)
end;

function plot_paccept_bysalience(df::DataFrame; color_gainsalient, color_lossalient, color_actual = RGBA{Float64}(0.5, 0.5, 0.5, 0.8), backend = "GLMakie")

    if backend == "GLMakie"
        GLMakie.activate!()
    elseif backend == "CairoMakie"
        CairoMakie.activate!()
    end

    fig = Figure(size = (1000, 550), fontsize = fontsize, fonts = (; regular = "Avenir Light"));
    xs = collect(-6:6);
    # Gain Salient Plot
    ax_gain = Axis(
        fig[1, 1], xlabel = "Value Difference\n(Gain - Loss)", ylabel = "Probability of\naccepting the gamble", 
        xgridvisible = false, ygridvisible = false,
        limits = ((-7, 7), (-0.1, 1.1)),
        xticks = -6:3:6,
        yticks = 0:0.25:1,
        
        xtrimspine = true, ytrimspine = true,
        # remve top and right spines
        topspinevisible = false, rightspinevisible = false,
    );
    lines!(ax_gain, xs, df.p_accept_simulated_gainsalient, color = color_gainsalient, linewidth = 10, alpha = 0.6); # Keep simulated line
    errorbars!(ax_gain, xs, df.p_accept_actual_gainsalient, df.sem_accept_actual_gainsalient, color = color_actual, whiskerwidth = 10); # Add error bars
    scatter!(ax_gain, xs, df.p_accept_actual_gainsalient, color = color_actual, markersize = 10, marker = :circle); # Keep scatter points
    vlines!(ax_gain, 0, color = :black, linestyle = (:dash, :loose), linewidth = 0.5)
    hlines!(ax_gain, 0.5, color = :black, linestyle = (:dash, :loose), linewidth = 0.5)

    # Loss Salient Plot
    ax_loss = Axis(
        fig[1, 2], xlabel = "Value Difference\n(Gain - Loss)", ylabel = "\n", 
        xgridvisible = false, ygridvisible = false,
        limits = ((-7, 7), (-0.1, 1.1)),
        xticks = -6:3:6,
        yticks = 0:0.25:1,
        
        xtrimspine = true, ytrimspine = true,
        # remve top and right spines
        topspinevisible = false, rightspinevisible = false,
    );
    lines!(ax_loss, xs, df.p_accept_simulated_lossalient, color = color_lossalient, linewidth = 10, alpha = 0.6); # Keep simulated line
    errorbars!(ax_loss, xs, df.p_accept_actual_lossalient, df.sem_accept_actual_lossalient, color = color_actual, whiskerwidth = 10); # Add error bars
    scatter!(ax_loss, xs, df.p_accept_actual_lossalient, color = color_actual, markersize = 10, marker = :circle); # Keep scatter points
    vlines!(ax_loss, 0, color = :black, linestyle = (:dash, :loose), linewidth = 0.5)
    hlines!(ax_loss, 0.5, color = :black, linestyle = (:dash, :loose), linewidth = 0.5)

    if string(Makie.current_backend()) == "CairoMakie"
        println("Current backend is CairoMakie, the plot will not be displayed.")
        return fig
    else
        display(fig)
        return fig
    end
end;

# ====== Brightness ======
# Parameters for the specific data to load
activation_base_dir = joinpath(
    config["local"]["data"], "results", "vam", "predictions"
);
checkpoint = 69;
train_actual, train_simulated, val_actual, val_simulated = load_predictions(activation_base_dir, checkpoint);

# RTs -------
fig = plot_rts(
    train_actual, train_simulated; 
    color_gainsalient = blend_colors(config["colors"]["gain-salient"]; alpha = 0.6), 
    color_lossalient = blend_colors(config["colors"]["loss-salient"]; alpha = 0.6), 
    color_actual = RGBA{Float64}(0.5, 0.5, 0.5, 0.4), 
    backend = "GLMakie"
);

# save the figure
fig = plot_rts(
    train_actual, train_simulated; 
    color_gainsalient = blend_colors(config["colors"]["gain-salient"]; alpha = 0.6), 
    color_lossalient = blend_colors(config["colors"]["loss-salient"]; alpha = 0.6), 
    color_actual = RGBA{Float64}(0.5, 0.5, 0.5, 0.4), 
    backend = "CairoMakie"
);

CairoMakie.save("figures/vam/predictions_rts.svg", fig);
CairoMakie.save("figures/vam/predictions_rts.png", fig, px_per_unit = 4);

# P(Accept) -------
df = get_paccept_bysalience(train_actual, train_simulated);
fig = plot_paccept_bysalience( df; 
    color_gainsalient = config["colors"]["gain-salient"], 
    color_lossalient = config["colors"]["loss-salient"], 
    color_actual = RGBA{Float64}(0.5, 0.5, 0.5, 0.7), 
    backend = "GLMakie"
);

# save the figure
fig = plot_paccept_bysalience( df; 
    color_gainsalient = config["colors"]["gain-salient"], 
    color_lossalient = config["colors"]["loss-salient"], 
    color_actual = RGBA{Float64}(0.5, 0.5, 0.5, 0.7), 
    backend = "CairoMakie"
);
CairoMakie.save("figures/vam/predictions_paccept.svg", fig, px_per_unit = 4);
CairoMakie.save("figures/vam/predictions_paccept.png", fig, px_per_unit = 4);

# ========================
