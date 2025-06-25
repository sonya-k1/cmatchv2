import streamlit as st
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import colorsys
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from statistics import geometric_mean
import re



def plot_bricks(path, part_colors):
    """
    Plots the matched parts as colored bricks.

    Args:
        path (list): A list of dictionaries, where each dictionary represents a matched part
                     and contains 'name', 'start', 'length', 'end', and 'score'.
        part_colors (dict): A dictionary mapping unique part names to colors.

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    y_pos = 0.5

    # Iterate through the path to draw bricks
    for i, part in enumerate(path):
        start = part['start']
        length_theoretical = part['length']
        name = part['name']
        score = part['score']
        end = part['end']
        length_observed = end - start

        color = part_colors.get(name, 'grey')

        rect = patches.Rectangle((start, y_pos), length_observed, 0.5, edgecolor='black', facecolor=color, alpha=0.5)

        ax.add_patch(rect)

        # Stagger label positions vertically
        label_y_pos = y_pos + 1.0 + (i % 3) * 0.5  # Adjusted stagger for readability

        ax.annotate(
            f'{name}\n({score:.2f})',
            xy=(start + length_observed / 2, y_pos + 0.25),
            xytext=(start + length_observed / 2, label_y_pos),
            fontsize=7,
            ha='left',
            va='center',
            color=color,
            arrowprops=dict(arrowstyle='-', color='black', lw=0.5)
        )

    handles = [patches.Patch(color=col, label=name) for name, col in part_colors.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=10, title="Part Names")

    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set x-axis limit to the maximum 'end' value
    if path:
        max_end = max(part['end'] for part in path)
        ax.set_xlim(-20, max_end + 100)
    else:
        ax.set_xlim(-20, 100) # Default if no path

    ax.set_ylim(0, 3 + len(path) * 0.5)  # Increase y-limit for spacing
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Part names and match scores", fontsize=12)
    ax.set_yticks([])  # Hide y-axis ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # plt.tight_layout()

    return fig

def visualise_parts(json_output_data: str):
    st.title('cMatch reconstruction map')
    data = json.loads(json_output_data)[0]
    len_data = len(data)

    unique_part_names = set()
    for result in data:
        if 'path' in result and result['path']:
            for part in result['path']:
                unique_part_names.add(part['name'])

    distinct_colors = plt.cm.get_cmap('viridis', len(unique_part_names))
    part_colors = {name: distinct_colors(i) for i, name in enumerate(sorted(list(unique_part_names)))}

    i = 1
    for result in data:
        
        # breakpoint()
        if 'path' in result and result['path']:
            path = result['path']
            fig = plot_bricks(path, part_colors)  
            score = result['score']
            label = result['reconstruct']
            target_name = result['target']

            st.text(f"Seq: {i}/{len_data} \n Target name: {target_name} \n cMatch Reconstruction Result: {label} \n Score: {score}")
            st.pyplot(fig)

        else:
            score = result['score']
            error = result.get('errors', 'Could not match all parts with this threshold')
            target_name = result['target']

            st.text(f"Seq: {i}/{len_data} \n Target name: {target_name} \n cMatch Reconstruction Result: {error} \n Score: {score}")
            # st.error('Could not match all parts with this threshold')
        i += 1


def extract_accuracy(target_name, file_prefix):
    """ 
    Extracts accuracy from filenames in the format: 'file_prefix_{accuracy}'.
    """
    match = re.search(rf'{file_prefix}_(\d+)_', target_name)
    return int(match.group(1)) if match else None

def visualise_distribution(result_json: str, overlap_pct, file_prefix=None, accuracy = None, plot_individual_parts=True, plot_over_acc_levels=True, ):
    '''
    Visualises distribution of part scores - will also plot across different accuracies if file_prefix provided
    file_prefix
    '''
    # st.title("Visualisation of multiple cMatch Scores")
    data = json.loads(result_json)
    
    records = []
    part_scores = []
    
    for entry in data:
        target = entry["target"]
        overall_score = entry["score"]
        
        records.append({
            "target": target,
            "overall_score": overall_score,
            "accuracy": accuracy if accuracy else extract_accuracy(target, file_prefix)
        })
        
        # Extract part scores
        if entry['path'] is not None:
            for part in entry.get("path", []):
                part_scores.append({
                    "target": target,
                    "part_name": part["name"],
                    "part_score": part["score"],
                    "accuracy": accuracy
                })
    
    df = pd.DataFrame(records)
    df_parts = pd.DataFrame(part_scores)
    
    df.sort_values(by=['overall_score'], inplace=True)
    df_unique = df.drop_duplicates()
    
    if df.empty:
        st.error("No valid data available for visualization.")
        return
    
    # Plot Overall Score Histograms for each Accuracy Level
    for accuracy in sorted(df["accuracy"].unique()):
        df_acc = df[df["accuracy"] == accuracy]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_acc['overall_score'], bins=100, kde=False, color='blue', edgecolor='black', ax=ax)
        
        # Calculate and plot mean/median
        mean_val = df_acc['overall_score'].mean()
        median_val = df_acc['overall_score'].median()
        ax.axvline(mean_val, color='red', linestyle='dashed', label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color='orange', linestyle='dashed', label=f"Median: {median_val:.2f}")
        ax.legend()
        
        ax.set_xlabel("cMatch Score")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Frequency Distribution of Overall Scores (Accuracy {accuracy})")
        ax.set_xlim([0, 1.1])
        ax.set_ylim([0, 50])
        ax.set_xticks([i/10 for i in range(11)]) 
        ax.set_yticks([i for i in range(0, 51, 5)])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        st.pyplot(fig)
    
    # Plot Part Score Histograms for each Accuracy Level
    if plot_individual_parts:
        if not df_parts.empty:
            for accuracy in sorted(df_parts["accuracy"].unique()):
                df_acc_parts = df_parts[df_parts["accuracy"] == accuracy]
                for part_name in df_acc_parts["part_name"].unique():
                    part_df = df_acc_parts[df_acc_parts["part_name"] == part_name]
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(part_df['part_score'], bins=50, kde=False, color='green', edgecolor='black', ax=ax)
                    
                    # Calculate and plot mean/median
                    mean_val = part_df['part_score'].mean()
                    median_val = part_df['part_score'].median()
                    ax.axvline(mean_val, color='red', linestyle='dashed', label=f"Mean: {mean_val:.2f}")
                    ax.axvline(median_val, color='orange', linestyle='dashed', label=f"Median: {median_val:.2f}")
                    ax.legend()
                    
                    ax.set_xlabel("Part Score")
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Frequency Distribution of {part_name} Scores (Accuracy {accuracy})")
                    ax.set_xlim([0, 1.1])
                    ax.set_ylim([0, 50])
                    ax.set_xticks([i/10 for i in range(11)]) 
                    ax.set_yticks([i for i in range(0, 51, 5)])
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                    st.pyplot(fig)
    
    if plot_over_acc_levels:
        # Plot Mean/Median for Overall Scores across Accuracy Levels
        overall_stats = df.groupby("accuracy")["overall_score"].agg(["mean", "median"]).reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(overall_stats["accuracy"], overall_stats["mean"], marker='o', linestyle='-', color='red', label='Mean')
        ax.plot(overall_stats["accuracy"], overall_stats["median"], marker='o', linestyle='-', color='orange', label='Median')
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Score")
        ax.set_title("Overall Score Mean & Median Across Accuracy Levels")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        st.pyplot(fig)
        
        # Plot Mean/Median for Each Part Score across Accuracy Levels
        if plot_individual_parts:
            if not df_parts.empty:
                for part_name in df_parts["part_name"].unique():
                    part_stats = df_parts[df_parts["part_name"] == part_name].groupby("accuracy")["part_score"].agg(["mean", "median"]).reset_index()
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(part_stats["accuracy"], part_stats["mean"], marker='o', linestyle='-', color='red', label='Mean')
                    ax.plot(part_stats["accuracy"], part_stats["median"], marker='o', linestyle='-', color='orange', label='Median')
                    ax.set_xlabel("Accuracy")
                    ax.set_ylabel("Score")
                    ax.set_title(f"{part_name} Score Mean & Median Across Accuracy Levels")
                    ax.legend()
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                    st.pyplot(fig)
    
    # st.text(f'Overlap tolerance: {overlap_pct}%')

 
def plot_error_types(errors_json, overlap_pct):
    """
    Function to parse errors, count occurrences of each error type, and plot a bar chart using Seaborn and Streamlit.
    
    Args:
        errors_json (str): JSON string of error data.

    """
    errors_list = json.loads(errors_json)

    error_messages = [error['errors'] for sublist in errors_list for error in sublist]

    
    df = pd.DataFrame({'Error Type': error_messages})
    error_counts = df['Error Type'].value_counts().reset_index()
    error_counts.columns = ['Error Type', 'Count']

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x="Count", y="Error Type", data=error_counts, palette="viridis")
    bar_plot.set_title(f"Error Types and Their Frequencies at Overlap tolerance of {overlap_pct} bases", fontsize=16)
    # bar_plot.set_title(f"Error Types and Their Frequencies", fontsize=16)
    bar_plot.set_xlabel("Count", fontsize=12)
    bar_plot.set_ylabel("Error Type", fontsize=12)
    st.pyplot(plt)


def plot_3d_multiple_scores(data:str, overlap_pct:int, threshold:float):
    """
    3D plot of cMatch scores for 3-part constructs only
    """

    # st.title("3D Plot of Genetic Part Scores")

    results = json.loads(data)[0]
    unique_acc_values = set()  

    for result in results:
        target_name = result['target']
        # Extract accuracy values from target names
        acc_value = target_name.split('_')[2]  
        unique_acc_values.add(acc_value)
            
    
    color_palette = cm.get_cmap('tab10')
    acc_color_map = {acc: mcolors.to_hex(color_palette(i / len(unique_acc_values))) 
                     for i, acc in enumerate(unique_acc_values)}

    # Lists to store 3D coordinates and hover info
    x_vals, y_vals, z_vals = [], [], []
    failed_part_x, failed_part_y, failed_part_z = [], [], []
    failed_part_hover_texts = []
    hover_texts = []
    point_colors = [] 
    failed_part_colours = []
    failed_overlap_x, failed_overlap_y, failed_overlap_z = [], [], []
    failed_overlap_hover_texts = []
    failed_overlap_colours = []
    for result in results:
        # breakpoint()
        target_name = result['target']
        acc_value = target_name.split('_')[2] 


        if result['path'] is not None and result.get('errors') is None:
            path = result['path']
            color = acc_color_map[acc_value]
            if result.get('errors') is None:
                color = 'green'  
            # elif result.get('errors') == 'Bases Overlapping or order is wrong':
            #     color = 'red'  
            # else:
            #     color = 'gray'  

            x = path[0]['score']
            y = path[1]['score']
            
            try:
                z = path[2]['score'] 
            except:
                breakpoint()
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
            point_colors.append(color)
            hover_texts.append(
                f"Target: {target_name}<br>"
                f"PBSim Accuracy: {acc_value}<br>"
                f"cMatch Score: {result['score']}<br>"
                f"Mean Score: {geometric_mean([x, y, z])}<br>"
                f"Score (Part 1): {x:.2f}<br>"
                f"Score (Part 2): {y:.2f}<br>"
                f"Score (Part 3): {z:.2f}<br>"
                f"Error: {result.get('errors', 'None')}"
            )

        else:
            # breakpoint()
            if result['path'] is not None and result.get('errors')=='Parts Overlapping or order is wrong':
                color='blue'
                path = result.get('path', [])
                scores = [0, 0, 0]  
                for i, part in enumerate(path[:3]):  
                    scores[i] = part['score']

                x = scores[0]  
                y = scores[1]  
                z = scores[2]  

                failed_overlap_x.append(x)
                failed_overlap_y.append(y)
                failed_overlap_z.append(z)
                failed_overlap_colours.append(color)
                failed_overlap_hover_texts.append(
                    f"Target: {target_name}<br>"
                    f"PBSim Accuracy: {acc_value}<br>"
                    f"cMatch Score: {result['score']}<br>"
                    f"Combined Score (Part 1): {x:.2f}<br>"
                    f"Score (Part 3): {y:.2f}<br>"
                    f"Score (Part 4): {z:.2f}<br>"
                    f"Error: {result.get('errors', 'None')}"
                )
            
            elif result['path'] is None:
                color = acc_color_map.get(acc_value, "#808080")
                color = 'red'
                failed_part_x.append(0 + np.random.uniform(0, 0.05))
                failed_part_y.append(0 + np.random.uniform(0, 0.05))
                failed_part_z.append(0 + np.random.uniform(0, 0.05))
                failed_part_colours.append(color)
                failed_part_hover_texts.append(
                    f"Target: {target_name}<br>Error: {result['errors']}"
                )


    fig = go.Figure()

   
    fig.add_trace(go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='markers',
        # text=[f"Point {i+1}" for i in range(len(x_vals))],
        hovertext=hover_texts,
        hoverinfo="text",
        marker=dict(
            size=5,
            color="#009E73" ,
            opacity=0.8
        ),
        name='Successful Reconstructions',
        showlegend=True
    ))


    fig.add_trace(go.Scatter3d(
        x=[1], y=[1], z=[1],
        mode='markers',
        text=["(1, 1, 1)"],
        hovertext="Reference Point: (1, 1, 1)",
        hoverinfo="text",
        marker=dict(
            size=10,
            color="#0072B2",  
            opacity=0.8
        ),
        name="Reference Point (1,1,1)"
    ))
    fig.add_trace(go.Scatter3d(
    x=failed_part_x, y=failed_part_y, z=failed_part_z,
    mode='markers',
    hovertext=failed_part_hover_texts,
    hoverinfo="text",
    marker=dict(
        size=5,
        color="#673AB7",  
        opacity=0.6  
    ),
    name=f'Failed on part matching with threshold {threshold}',
    showlegend=True
))
    fig.add_trace(go.Scatter3d(
    x=failed_overlap_x, y=failed_overlap_y, z=failed_overlap_z,
    mode='markers',
    hovertext=failed_overlap_hover_texts,
    hoverinfo="text",
    marker=dict(
        size=5,
        color="#CC79A7",  
        opacity=0.8  
    ),
    # name=f'Failed on reconstruction with base overlap tolerance {overlap_pct}%',
    name=f'Failed on reconstruction',
    showlegend=True
))


    fig.update_layout(
        scene=dict( 
            xaxis_title="Score of combined parts (Part 1 & 2)",
            yaxis_title="Score (Part 3)",
            zaxis_title="Score (Part 4)",
            xaxis=dict(
            range=[0, 1],  
            tickmode="array",
            tickvals=[i / 10 for i in range(0, 11)],  
            ticktext=[f"{i/10:.1f}" for i in range(0, 11)]
        ),
        yaxis=dict(
            range=[0, 1],
            tickmode="array",
            tickvals=[i / 10 for i in range(0, 11)],
            ticktext=[f"{i/10:.1f}" for i in range(0, 11)]
        ),
        zaxis=dict(
            range=[0, 1],
            tickmode="array",
            tickvals=[i / 10 for i in range(0, 11)],
            ticktext=[f"{i/10:.1f}" for i in range(0, 11)]
        ),
            aspectmode="cube",
               
        camera=dict(
            eye=dict(x=1.4, y=-1.4, z=1.4)  # Adjust to the midpoint between (0,0,0) and (1,1,1)
        )
        ),
        showlegend=True,
        autosize=False,
        width=1000,  
        height=700,  
        margin=dict(l=0, r=0, b=50, t=50),
    )
    threshold_line_color = "red"

    fig.add_trace(go.Scatter3d(
        x=[threshold, threshold], 
        y=[0, 1],  
        z=[threshold, threshold],  
        mode='lines',
        line=dict(color=threshold_line_color, width=3, dash='dash'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[threshold, threshold],  
        y=[threshold, threshold], 
        z=[0, 1],  
        mode='lines',
        line=dict(color=threshold_line_color, width=3, dash='dash'),
        showlegend=False

    ))

    fig.add_trace(go.Scatter3d(
        x=[0, 1],  
        y=[threshold, threshold],  
        z=[threshold, threshold], 
        mode='lines',
        line=dict(color=threshold_line_color, width=3, dash='dash'),
        showlegend=False

    ))

    plane_color = "rgba(255, 0, 0, 0.2)"  

    fig.add_trace(go.Mesh3d(
        x=[threshold, threshold, threshold, threshold],  
        y=[0, 1, 1, 0],  
        z=[0, 0, 1, 1],  
        i=[0, 1, 2, 0],  
        j=[1, 2, 3, 3],  
        k=[2, 3, 0, 1],  
        color=plane_color,
        opacity=0.1,
        name=f"Plane at x = {threshold}"
    ))

    fig.add_trace(go.Mesh3d(
        x=[0, 1, 1, 0],  
        y=[threshold, threshold, threshold, threshold],  
        z=[0, 0, 1, 1],  
        i=[0, 1, 2, 0],  
        j=[1, 2, 3, 3],  
        k=[2, 3, 0, 1],  
        color=plane_color,
        opacity=0.1,
        name=f"Plane at y = {threshold}"
    ))

    fig.add_trace(go.Mesh3d(
        x=[0, 1, 1, 0],  
        y=[0, 0, 1, 1],  
        z=[threshold, threshold, threshold, threshold],  
        i=[0, 1, 2, 0],  
        j=[1, 2, 3, 3],  
        k=[2, 3, 0, 1],  
        color=plane_color,
        opacity=0.1,
        name=f"Plane at z = {threshold}"
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.text(f'Number of constructs failing the similarity threshold {threshold} for any of the parts: {len(failed_part_x)}/{len(results)} \n'
            f'Number of constructs failing reconstruction using overlap tolerance of {overlap_pct}%: {len(failed_overlap_x)}/{len(results)} \n'
            f'Number of successful reconstructions: {len(x_vals)}/{len(results)}')

