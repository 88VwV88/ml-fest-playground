import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import time

# Set the style for all plots
plt.style.use('fivethirtyeight')
sns.set_palette("pastel")

# Page configuration
st.set_page_config(
    page_title="ML Visualization Playground",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    h1, h2, h3 {margin-bottom: 0.5rem;}
    .stSlider {padding: 1rem 0;}
    .stExpander {border-radius: 8px;}
    .css-18e3th9 {padding-top: 1rem;}
    .css-1d391kg {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("ML Visualization Playground")
page = st.sidebar.radio(
    "Choose a visualization:",
    ["Normal Distribution", "Central Limit Theorem",
        "Linear Regression", "Polynomial Regression with SGD"]
)

if page == "Normal Distribution":
    st.title("Normal Distribution Explorer")
    st.write("""
    Explore how changing the mean and standard deviation affects the normal distribution curve.
    The normal distribution is a continuous probability distribution that is symmetric about the mean.
    """)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Parameters")
        mean = st.slider("Mean (μ)", -10.0, 10.0, 0.0, 0.1)
        std_dev = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1)

        with st.expander("Distribution Properties"):
            st.write(f"**Mean:** {mean}")
            st.write(f"**Standard Deviation:** {std_dev}")
            st.write(f"**Variance:** {std_dev**2}")

            # Calculate probabilities for common ranges
            within_1_std = 2 * stats.norm.cdf(1) - 1
            within_2_std = 2 * stats.norm.cdf(2) - 1
            within_3_std = 2 * stats.norm.cdf(3) - 1

            st.write("### Probability Ranges")
            st.write(f"μ ± 1σ: {within_1_std:.2%}")
            st.write(f"μ ± 2σ: {within_2_std:.2%}")
            st.write(f"μ ± 3σ: {within_3_std:.2%}")

    with col1:
        # Generate normal distribution data
        x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
        y = stats.norm.pdf(x, mean, std_dev)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, 'b-', linewidth=2, label=f'N({mean}, {std_dev}²)')
        ax.set_xlim(-10, 10)

        # Shade areas within 1, 2, and 3 standard deviations
        ax.fill_between(x, y, where=(x >= mean-std_dev) & (x <= mean+std_dev),
                        color='blue', alpha=0.3, label='μ ± 1σ (68.27%)')
        ax.fill_between(x, y, where=(x >= mean-2*std_dev) & (x <= mean+2*std_dev) &
                        ((x < mean-std_dev) | (x > mean+std_dev)),
                        color='green', alpha=0.3, label='μ ± 2σ (95.45%)')
        ax.fill_between(x, y, where=(x >= mean-3*std_dev) & (x <= mean+3*std_dev) &
                        ((x < mean-2*std_dev) | (x > mean+2*std_dev)),
                        color='red', alpha=0.3, label='μ ± 3σ (99.73%)')

        # Add vertical lines for mean and standard deviations
        ax.axvline(mean, color='black', linestyle='--',
                   alpha=0.7, label='Mean')
        ax.axvline(mean + std_dev, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(mean - std_dev, color='gray', linestyle=':', alpha=0.7)

        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Normal Distribution - N({mean}, {std_dev}²)')
        ax.legend()

        # Display the plot
        st.pyplot(fig)

        # Add description
        st.info("""
        **Key Points:**
        - The normal distribution is completely defined by its mean and standard deviation
        - It's symmetric around the mean
        - The total area under the curve is 1 (100% probability)
        - Approximately 68% of data falls within ±1σ, 95% within ±2σ, and 99.7% within ±3σ
        """)

elif page == "Central Limit Theorem":
    st.title("Central Limit Theorem Explorer")
    st.write("""
    The Central Limit Theorem states that the sampling distribution of the sample mean approaches a normal distribution
    as the sample size gets larger, regardless of the shape of the population distribution.
    """)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Parameters")
        distribution_type = st.selectbox(
            "Select initial distribution",
            ["Uniform", "Exponential", "Bimodal", "Skewed", "Custom"]
        )

        sample_size = st.slider("Sample Size (n)", 1, 100, 30)
        num_samples = st.slider("Number of Samples", 100, 5000, 1000)

        if distribution_type == "Custom":
            custom_values = st.text_area(
                "Enter comma-separated values for your distribution")
            try:
                if custom_values:
                    custom_distribution = np.array(
                        [float(x) for x in custom_values.split(',')])
                else:
                    custom_distribution = np.random.normal(0, 1, 1000)
            except:
                st.error("Please enter valid comma-separated numbers")
                custom_distribution = np.random.normal(0, 1, 1000)

    with col1:
        # Generate the original distribution based on selection
        if distribution_type == "Uniform":
            population = np.random.uniform(-3, 3, 10000)
            theoretical_mean = 0
            # Std dev of uniform dist from -3 to 3
            theoretical_std = 6/np.sqrt(12)
        elif distribution_type == "Exponential":
            population = np.random.exponential(scale=1.0, size=10000)
            theoretical_mean = 1.0
            theoretical_std = 1.0
        elif distribution_type == "Bimodal":
            population = np.concatenate([
                np.random.normal(-2, 0.5, 5000),
                np.random.normal(2, 0.5, 5000)
            ])
            theoretical_mean = 0
            theoretical_std = np.std(population)
        elif distribution_type == "Skewed":
            population = np.random.gamma(shape=2, scale=1, size=10000)
            theoretical_mean = 2 * 1  # shape * scale
            theoretical_std = np.sqrt(2) * 1  # sqrt(shape) * scale
        elif distribution_type == "Custom":
            population = custom_distribution
            theoretical_mean = np.mean(population)
            theoretical_std = np.std(population)

        # Calculate the sampling distribution
        sample_means = []
        for _ in range(num_samples):
            sample = np.random.choice(
                population, size=sample_size, replace=True)
            sample_means.append(np.mean(sample))

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot the original distribution
        sns.histplot(population, kde=True, ax=ax1, color='skyblue')
        ax1.set_title(f'Original {distribution_type} Distribution')
        ax1.axvline(theoretical_mean, color='red', linestyle='--',
                    label=f'Mean = {theoretical_mean:.2f}')
        ax1.legend()

        # Plot the sampling distribution
        sns.histplot(sample_means, kde=True, ax=ax2, color='green')
        ax2.set_title(
            f'Sampling Distribution of the Mean (n={sample_size}, samples={num_samples})')

        # Calculate expected mean and standard error
        expected_mean = theoretical_mean
        expected_se = theoretical_std / np.sqrt(sample_size)

        # Overlay the theoretical normal curve
        x = np.linspace(min(sample_means), max(sample_means), 1000)
        y = stats.norm.pdf(x, expected_mean, expected_se)
        ax2.plot(x, y * (len(sample_means) * (max(sample_means) - min(sample_means)) / 10),
                 'r--', linewidth=2, label='Expected Normal Distribution')

        ax2.axvline(expected_mean, color='red', linestyle='--',
                    label=f'Expected Mean = {expected_mean:.2f}')
        ax2.axvline(np.mean(sample_means), color='green', linestyle='-',
                    label=f'Actual Mean = {np.mean(sample_means):.2f}')
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Display statistics
        st.subheader("Statistics")
        col_stats1, col_stats2 = st.columns(2)

        with col_stats1:
            st.write("**Original Distribution:**")
            st.write(f"Mean: {np.mean(population):.4f}")
            st.write(f"Standard Deviation: {np.std(population):.4f}")

        with col_stats2:
            st.write("**Sampling Distribution:**")
            st.write(f"Mean: {np.mean(sample_means):.4f}")
            st.write(f"Standard Deviation (SE): {np.std(sample_means):.4f}")
            st.write(f"Expected SE: {expected_se:.4f}")

        # Add explanation
        st.info("""
        **Central Limit Theorem in Action:**
        - As sample size increases, the sampling distribution of the mean approaches a normal distribution
        - The mean of the sampling distribution equals the population mean
        - The standard deviation of the sampling distribution (standard error) equals population_std/√n
        - This holds true regardless of the shape of the original distribution
        """)

elif page == "Linear Regression":
    st.title("Linear Regression Explorer")
    st.write("""
    Experiment with a single-feature linear regression model. 
    Adjust the weight and bias parameters to see how the model fits the data and how the loss changes.
    """)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Dataset Parameters")
        n_points = st.slider("Number of data points", 10, 100, 30)
        noise_level = st.slider("Noise level", 0.0, 5.0, 1.0, 0.1)
        true_weight = st.slider("True weight", -5.0, 5.0, 2.0, 0.1)
        true_bias = st.slider("True bias", -5.0, 5.0, 1.0, 0.1)

        st.subheader("Model Parameters")
        weight = st.slider("Weight (w)", -10.0, 10.0, 0.0, 0.1)
        bias = st.slider("Bias (b)", -10.0, 10.0, 0.0, 0.1)

        # Generate random data
        np.random.seed(42)  # For reproducibility
        X = np.random.uniform(-5, 5, n_points)
        y = true_weight * X + true_bias + \
            np.random.normal(0, noise_level, n_points)

        # Calculate loss for current parameters
        predictions = weight * X + bias
        mse_loss = np.mean((predictions - y) ** 2)
        mae_loss = np.mean(np.abs(predictions - y))

        st.subheader("Loss Metrics")
        st.metric("Mean Squared Error", f"{mse_loss:.4f}")
        st.metric("Mean Absolute Error", f"{mae_loss:.4f}")

        # Loss surface visualization
        show_loss_surface = st.checkbox("Show Loss Surface", value=True)

    with col1:
        # Create figure for regression line
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        # Plot data points
        ax1.scatter(X, y, color='blue', alpha=0.6, label='Data points')

        # Plot the true line
        x_line = np.linspace(-6, 6, 100)
        y_true = true_weight * x_line + true_bias
        ax1.plot(x_line, y_true, 'g-', linewidth=2,
                 label=f'True: y = {true_weight}x + {true_bias}')

        # Plot the model line
        y_pred = weight * x_line + bias
        ax1.plot(x_line, y_pred, 'r--', linewidth=2,
                 label=f'Model: y = {weight}x + {bias}')

        # Plot the residuals
        for i in range(len(X)):
            ax1.plot([X[i], X[i]], [y[i], weight *
                     X[i] + bias], 'k-', alpha=0.2)

        ax1.set_xlabel('X')
        ax1.set_ylabel('y')
        ax1.set_title('Linear Regression: Data vs Model')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Set reasonable axis limits
        ax1.set_xlim(-6, 6)
        y_min = min(min(y), min(y_true), min(y_pred)) - 1
        y_max = max(max(y), max(y_true), max(y_pred)) + 1
        ax1.set_ylim(y_min, y_max)

        st.pyplot(fig1)

        # Show loss surface if checkbox is selected
        if show_loss_surface:
            # Create weight and bias grid for loss surface
            w_range = np.linspace(true_weight - 5, true_weight + 5, 50)
            b_range = np.linspace(true_bias - 5, true_bias + 5, 50)
            W, B = np.meshgrid(w_range, b_range)
            Z = np.zeros_like(W)

            # Calculate MSE for each weight-bias combination
            for i in range(len(w_range)):
                for j in range(len(b_range)):
                    w_temp = w_range[i]
                    b_temp = b_range[j]
                    y_pred_temp = w_temp * X + b_temp
                    Z[j, i] = np.mean((y_pred_temp - y) ** 2)

            # Create the loss surface plot using plotly
            true_mse = np.mean(((true_weight * X + true_bias) - y) ** 2)

            fig2 = go.Figure()

            # Add surface plot
            fig2.add_trace(
                go.Surface(
                    x=W, y=B, z=Z,
                    colorscale='viridis',
                    opacity=0.8,
                )
            )

            # Add current parameters point
            fig2.add_trace(
                go.Scatter3d(
                    x=[weight], y=[bias], z=[mse_loss],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name='Current parameters'
                )
            )

            # Add true parameters point
            fig2.add_trace(
                go.Scatter3d(
                    x=[true_weight], y=[true_bias], z=[true_mse],
                    mode='markers',
                    marker=dict(size=8, color='green'),
                    name='True parameters'
                )
            )

            # Update layout
            fig2.update_layout(
                title='Loss Surface',
                scene=dict(
                    xaxis_title='Weight (w)',
                    yaxis_title='Bias (b)',
                    zaxis_title='Mean Squared Error',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                ),
                width=700,
                height=600,
                margin=dict(l=0, r=0, b=0, t=40)
            )

            st.plotly_chart(fig2)

            # Create contour plot with plotly
            fig3 = go.Figure()

            # Add contour plot
            fig3.add_trace(
                go.Contour(
                    z=Z,
                    x=w_range,
                    y=b_range,
                    colorscale='viridis',
                    contours=dict(
                        showlabels=True,
                        labelfont=dict(size=12, color='white')
                    )
                )
            )

            # Add current parameters point
            fig3.add_trace(
                go.Scatter(
                    x=[weight], y=[bias],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='x'),
                    name='Current parameters'
                )
            )

            # Add true parameters point
            fig3.add_trace(
                go.Scatter(
                    x=[true_weight], y=[true_bias],
                    mode='markers',
                    marker=dict(size=12, color='green', symbol='star'),
                    name='True parameters'
                )
            )

            # Add horizontal and vertical lines for current parameters
            fig3.add_shape(
                type="line",
                x0=w_range[0], x1=w_range[-1],
                y0=bias, y1=bias,
                line=dict(color="red", width=1, dash="dash")
            )

            fig3.add_shape(
                type="line",
                x0=weight, x1=weight,
                y0=b_range[0], y1=b_range[-1],
                line=dict(color="red", width=1, dash="dash")
            )

            # Update layout
            fig3.update_layout(
                title='Loss Contour Map',
                xaxis_title='Weight (w)',
                yaxis_title='Bias (b)',
                width=700,
                height=500,
                showlegend=True
            )

            st.plotly_chart(fig3)

        # Add explanation
        st.info("""
        **Linear Regression Concept:**
        - The model predicts a value (y) based on a linear function of x: ŷ = wx + b
        - Weight (w) controls the slope of the line
        - Bias (b) controls the y-intercept
        - The loss function measures how well the model fits the data
        - Mean Squared Error (MSE): average of squared differences between predictions and actual values
        - Mean Absolute Error (MAE): average of absolute differences between predictions and actual values
        - The goal is to find the weight and bias that minimize the loss
        """)

elif page == "Polynomial Regression with SGD":
    st.title("Polynomial Regression with Stochastic Gradient Descent")
    st.write("""
    Visualize how polynomial regression fits curves to data and how Stochastic Gradient Descent (SGD) 
    optimizes the model parameters. Watch the convergence process in real-time.
    """)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Dataset Parameters")
        n_points = st.slider("Number of data points", 10, 200, 50)
        noise_level = st.slider("Noise level", 0.0, 2.0, 0.5, 0.1)
        true_function = st.selectbox(
            "True function",
            ["Quadratic", "Cubic", "Sinusoidal", "Exponential"]
        )

        st.subheader("Model Parameters")
        poly_degree = st.slider("Polynomial degree", 1, 10, 3)
        learning_rate = st.slider("Learning rate", 0.001, 0.5, 0.01, 0.001)
        n_iterations = st.slider("Number of iterations", 100, 5000, 1000, 100)
        batch_size = st.slider("Batch size", 1, min(50, n_points), 10)

        # Generate data based on selected function
        np.random.seed(42)
        X = np.random.uniform(-3, 3, n_points)

        if true_function == "Quadratic":
            y_true = 1.5 * X**2 - 0.5 * X + 1
        elif true_function == "Cubic":
            y_true = 0.5 * X**3 - 0.5 * X**2 + X
        elif true_function == "Sinusoidal":
            y_true = np.sin(2 * X) + 0.5 * X
        elif true_function == "Exponential":
            y_true = np.exp(0.5 * X) / 5

        y = y_true + np.random.normal(0, noise_level, n_points)

        # Button to start SGD
        start_sgd = st.button("Start SGD Training")

    with col1:
        # Function to create polynomial features
        def create_poly_features(X, degree):
            if isinstance(X, (int, float)):
                X = np.array([X])
            X_reshaped = X.reshape(-1, 1)
            X_poly = np.ones((len(X_reshaped), degree + 1))
            for i in range(1, degree + 1):
                X_poly[:, i] = X_reshaped.flatten() ** i
            return X_poly

        # Function for SGD
        def sgd_step(X_batch, y_batch, weights, lr):
            y_pred = X_batch @ weights
            error = y_pred - y_batch
            gradient = X_batch.T @ error / len(X_batch)
            new_weights = weights - lr * gradient
            return new_weights, np.mean(error ** 2)

        # Create polynomial features
        X_poly = create_poly_features(X, poly_degree)

        # Initialize weights
        weights = np.zeros(poly_degree + 1)

        # Setup the figure for visualization
        fig = go.Figure()

        # Plot original data points
        fig.add_trace(go.Scatter(
            x=X,
            y=y,
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Data points'
        ))

        # Plot the true function
        X_line = np.linspace(-3, 3, 100)
        if true_function == "Quadratic":
            y_line = 1.5 * X_line**2 - 0.5 * X_line + 1
        elif true_function == "Cubic":
            y_line = 0.5 * X_line**3 - 0.5 * X_line**2 + X_line
        elif true_function == "Sinusoidal":
            y_line = np.sin(2 * X_line) + 0.5 * X_line
        elif true_function == "Exponential":
            y_line = np.exp(0.5 * X_line) / 5

        fig.add_trace(go.Scatter(
            x=X_line,
            y=y_line,
            mode='lines',
            line=dict(color='green', width=2),
            name='True function'
        ))

        # Initial model line (before SGD)
        X_line_poly = create_poly_features(X_line, poly_degree)
        y_line_pred = X_line_poly @ weights

        model_line = fig.add_trace(go.Scatter(
            x=X_line,
            y=y_line_pred,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Model prediction'
        ))

        # Update layout
        fig.update_layout(
            title=f'Polynomial Regression (degree = {poly_degree})',
            xaxis_title='X',
            yaxis_title='y',
            height=500,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Create a placeholder for the figure
        chart_placeholder = st.plotly_chart(fig)

        # Create a placeholder for training progress
        progress_placeholder = st.empty()

        # Create a placeholder for the loss curve
        loss_chart_placeholder = st.empty()

        if start_sgd:
            # Setup the loss tracking
            losses = []

            # Create a progress bar
            progress_bar = st.progress(0)

            # Create a figure for the loss curve
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(
                x=[],
                y=[],
                mode='lines+markers',
                line=dict(color='purple', width=2),
                name='Training Loss'
            ))

            loss_fig.update_layout(
                title='Training Loss over Iterations',
                xaxis_title='Iteration',
                yaxis_title='Mean Squared Error',
                height=300,
                margin=dict(l=0, r=0, b=0, t=40)
            )

            # Run SGD
            indices = np.arange(len(X))
            n_batches = max(1, len(X) // batch_size)

            for i in range(1, n_iterations + 1):
                # Shuffle data for each epoch
                if i % n_batches == 1:
                    np.random.shuffle(indices)

                # Get batch indices
                batch_start = ((i-1) % n_batches) * batch_size
                batch_end = min(batch_start + batch_size, len(X))
                batch_idx = indices[batch_start:batch_end]
                X_batch = X_poly[batch_idx]
                y_batch = y[batch_idx]

                # Update weights
                weights, batch_loss = sgd_step(
                    X_batch, y_batch, weights, learning_rate)

                # Calculate overall loss for tracking
                y_pred_all = X_poly @ weights
                loss = np.mean((y_pred_all - y) ** 2)
                losses.append(loss)

                # Update model line every few iterations
                if i % max(1, n_iterations // 100) == 0 or i == n_iterations:
                    y_line_pred = X_line_poly @ weights

                    # Update the plot
                    fig.data[2].y = y_line_pred
                    chart_placeholder.plotly_chart(fig)

                    # Update the loss curve
                    loss_fig.data[0].x = list(range(1, len(losses) + 1))
                    loss_fig.data[0].y = losses
                    loss_chart_placeholder.plotly_chart(loss_fig)

                    # Update progress
                    progress_text = f'Iteration {i}/{n_iterations} - Current Loss: {loss:.6f}'
                    progress_placeholder.text(progress_text)
                    progress_bar.progress(i / n_iterations)

                    # Slow down to show the training process
                    time.sleep(0.01)

            st.success(f"Training complete! Final loss: {losses[-1]:.6f}")

            # Display the final model equation
            equation = "y = "
            for i in range(poly_degree + 1):
                if i == 0:
                    equation += f"{weights[i]:.4f}"
                else:
                    if weights[i] >= 0:
                        equation += f" + {weights[i]:.4f}x^{i}"
                    else:
                        equation += f" - {abs(weights[i]):.4f}x^{i}"

            st.write(f"**Final Model Equation:** {equation}")

        # Add explanation
        st.info("""
        **Polynomial Regression and SGD:**
        - Polynomial regression extends linear regression by adding polynomial terms (x², x³, etc.)
        - Higher degrees can fit more complex patterns but may overfit the data
        - Stochastic Gradient Descent (SGD) is an optimization algorithm that:
          - Updates model parameters using small batches of data
          - Minimizes the loss function by taking steps in the direction of steepest descent
          - Uses a learning rate to control the size of these steps
          - Is more efficient than batch gradient descent for large datasets
        - Watch how the model improves as SGD iteratively refines the parameters!
        """)

st.sidebar.markdown("""
### How to Use
1. Select a visualization from the sidebar
2. Adjust parameters using the sliders
3. Observe how changes affect the visualization

### About
This app demonstrates key statistical and machine learning concepts:
- Normal Distribution
- Central Limit Theorem
- Linear Regression

Created with Streamlit, NumPy, Pandas, and Matplotlib.
""")

st.sidebar.info("© 2025 Machine Learning Visualization Playground")
