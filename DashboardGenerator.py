# -*- coding: utf-8 -*-
"""
Flexible Dashboard Generator with YAML Configuration
"""

import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages


class DashboardGenerator:
    """Create a PDF dashboard with flexible layout defined by config."""

    def __init__(self, config_path, output_filename, fig_size=(18, 13), dpi=100):
        """
        Args:
            config_path: Path to YAML configuration file
            output_filename: Output PDF filename
            fig_size: Figure size in inches
            dpi: Resolution
        """
        self.output_filename = output_filename
        self.fig_size = fig_size
        self.dpi = dpi

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._validate_config()
        
        if not output_filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF output is supported.")
        
        self.pdf = PdfPages(self.output_filename)
        print(f"--- PDF initialized: {self.output_filename} ---")

        # Store colorbars to remove them later
        self.colorbars = []

        # Create figure and layout
        self._setup_figure()

    def _validate_config(self):
        """Validate the configuration structure."""
        if 'layout' not in self.config:
            raise ValueError("Config must contain 'layout' section")
        
        valid_types = {'image', 'bitmap', 'graph'}
        for idx, row in enumerate(self.config['layout']):
            if 'components' not in row:
                raise ValueError(f"Row {idx} missing 'components' field")
            for comp in row['components']:
                if comp not in valid_types:
                    raise ValueError(f"Invalid component type '{comp}' in row {idx}. "
                                   f"Must be one of: {valid_types}")

    def _setup_figure(self):
        """Create the matplotlib figure based on configuration."""
        layout = self.config['layout']
        num_rows = len(layout)
        
        # Calculate height ratios
        height_ratios = []
        for row in layout:
            # Images get more height
            if 'image' in row['components']:
                height_ratios.append(1.0)
            else:
                height_ratios.append(0.75)
        
        # Determine max columns needed
        max_cols = max(len(row['components']) for row in layout)
        
        # Create figure
        self.fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        self.fig.set_tight_layout(False)
        
        # Create grid
        gs = self.fig.add_gridspec(
            nrows=num_rows,
            ncols=max_cols,
            height_ratios=height_ratios,
            wspace=0.04,
            hspace=0.2,
        )
        
        # Create axes for each component
        self.axes = []
        for row_idx, row in enumerate(layout):
            row_axes = []
            components = row['components']
            
            for col_idx, comp_type in enumerate(components):
                # Images can span multiple columns
                if comp_type == 'image' and row.get('image_span', 1) > 1:
                    span = row['image_span']
                    ax = self.fig.add_subplot(gs[row_idx, col_idx:col_idx+span])
                else:
                    ax = self.fig.add_subplot(gs[row_idx, col_idx])
                
                row_axes.append({
                    'ax': ax,
                    'type': comp_type
                })
            
            self.axes.append(row_axes)

    def _clear_axes(self):
        """Clear all axes before drawing new page."""
        # Remove previous colorbars
        for cbar in self.colorbars:
            cbar.remove()
        self.colorbars = []
        
        # Clear axes
        for row in self.axes:
            for item in row:
                item['ax'].clear()

    def _draw_image(self, ax, data):
        """Draw an image component."""
        img = data['content'].copy().astype(float)
        cmap = 'gray' if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(data['title'])
        ax.axis('off')

    def _draw_bitmap(self, ax, data):
        """Draw a bitmap component."""
        im = ax.imshow(data['content'], cmap='viridis')
        ax.set_title(data['title'])
        ax.axis('off')
        # Add colorbar and store reference
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.colorbars.append(cbar)

    def _draw_graph(self, ax, data):
        """
        Draw a graph component with optional thresholds, without connecting lines
        between separated threshold segments (line breaks at NaNs).
        """
        plots = data['content']['plots']
        background_color = data.get('background_color', None)
        
        # Apply background color to entire axis
        if background_color is not None:
            ax.set_facecolor(background_color)
    
        for plot_data in plots:
            x = np.asarray(plot_data['x'])
            y = np.asarray(plot_data['y'], dtype=float)  # dtype=float allows NaNs
    
            color = plot_data.get('color', 'tab:blue')
            threshold = plot_data.get('threshold', None)
            threshold_condition = plot_data.get('threshold_condition', 'above')
    
            base_label = plot_data.get('label', f'{color} plot')
            thr_label = plot_data.get('threshold_label', f'{color} threshold')
    
            if threshold is not None:
                if threshold_condition == 'above':
                    bold_mask = y > threshold
                else:  # 'below'
                    bold_mask = y < threshold
    
                # Normal segment(s): set bold points to NaN to break the line
                y_normal = y.copy()
                y_normal[bold_mask] = np.nan
                ax.plot(
                    x, y_normal, 'o-',
                    color=color, linewidth=1.5, markersize=3,
                    label=base_label
                )
    
                # Bold segment(s): set non-bold points to NaN to break the line
                y_bold = y.copy()
                y_bold[~bold_mask] = np.nan
                ax.plot(
                    x, y_bold, 'o-',
                    color=color, linewidth=3, markersize=6,
                    label='_nolegend_'
                )
    
                # Threshold line
                ax.axhline(
                    y=threshold, color=color, linestyle='--',
                    linewidth=2, alpha=0.7, label=thr_label
                )
            else:
                ax.plot(
                    x, y, 'o-',
                    color=color, linewidth=2, markersize=3,
                    label=base_label
                )
    
        ax.set_title(data['title'])
        ax.grid(True, alpha=0.3)
    
        if len(plots) > 1 or any(p.get('threshold') is not None for p in plots):
            ax.legend(loc='best', fontsize=8)

    def add_page(self, page_data):
        """
        Add a page to the PDF.
        
        Args:
            page_data: List of rows, where each row is a list of dicts with:
                      - For images/bitmaps: {'title': str, 'content': array}
                      - For graphs: {'title': str, 'content': {'plots': [...]}}
        """
        if len(page_data) != len(self.axes):
            raise ValueError(f"Expected {len(self.axes)} rows, got {len(page_data)}")
        
        self._clear_axes()
        
        for row_idx, (row_axes, row_data) in enumerate(zip(self.axes, page_data)):
            if len(row_data) != len(row_axes):
                raise ValueError(
                    f"Row {row_idx}: expected {len(row_axes)} components, "
                    f"got {len(row_data)}"
                )
            
            for col_idx, (ax_info, data) in enumerate(zip(row_axes, row_data)):
                comp_type = ax_info['type']
                ax = ax_info['ax']
                
                if comp_type == 'image':
                    self._draw_image(ax, data)
                elif comp_type == 'bitmap':
                    self._draw_bitmap(ax, data)
                elif comp_type == 'graph':
                    self._draw_graph(ax, data)
        
        self.pdf.savefig(self.fig)
        print("PDF page added.")

    def close(self):
        """Close and save the PDF."""
        self.pdf.close()
        plt.close(self.fig)
        print(f"PDF saved to {self.output_filename}")


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    NUM_PAGES = 4
    
    dashboard = DashboardGenerator(
        config_path='dashboard_layout.yaml',
        output_filename='flexible_dashboard - 4.pdf'
    )
    
    for page_idx in range(NUM_PAGES):
        # Row 1: [image, image, bitmap, bitmap]
        row1_data = []
        # First image
        img1 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        row1_data.append({'title': f'Image 1 - Page {page_idx+1}', 'content': img1})
        
        # Second image
        img2 = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        row1_data.append({'title': f'Image 2 - Page {page_idx+1}', 'content': img2})
        
        # Two bitmaps
        for i in range(2):
            bmp = np.zeros((80, 80), dtype=np.float32)
            cx = 40 + int(10 * np.cos(page_idx + i))
            cy = 40 + int(10 * np.sin(page_idx + i))
            cv2.circle(bmp, (cx, cy), 10 + i*2, 1.0, -1)
            bmp = cv2.GaussianBlur(bmp, (7, 7), 1.5)
            row1_data.append({'title': f'Bitmap 1.{i+1} - Page {page_idx+1}', 'content': bmp})
        
        # Row 2: [bitmap, bitmap, bitmap, bitmap]
        row2_data = []
        for i in range(4):
            bmp = np.random.rand(80, 80) * (i + 1) / 4
            row2_data.append({'title': f'Bitmap 2.{i+1} - Page {page_idx+1}', 'content': bmp})
        
        # Row 3: [graph, graph, graph, graph]
        # Mix of Type 1 (dual plot) and Type 2 (single red plot)
        row3_data = []
        x = np.linspace(0, 10, 200)
        
        # Graph 1: Type 1 - Red and Blue plots with thresholds
        y_red = np.sin(x + page_idx * 0.5) * 2 + 3
        y_blue = np.cos(x + page_idx * 0.3) * 1.5 + 2
        row3_data.append({
            'title': f'Dual Plot - Page {page_idx+1}',
            'background_color': 'lightcoral',
            'content': {
                'plots': [
                    {
                        'x': x,
                        'y': y_red,
                        'color': 'red',
                        'threshold': 3.5,
                        'threshold_condition': 'above'
                    },
                    {
                        'x': x,
                        'y': y_blue,
                        'color': 'blue',
                        'threshold': 1.5,
                        'threshold_condition': 'below'
                    }
                ]
            }
        })
        
        # Graph 2: Type 2 - Single red plot with threshold
        y_red2 = np.sin(x * 2 + page_idx) * 1.5 + 2.5

        if page_idx % 2 == 0:
            bg_color = 'lightcoral'
        else:
            bg_color = 'white'



        row3_data.append({
            'title': f'Single Red Plot - Page {page_idx+1}',
            'background_color': bg_color,

            'content': {
                'plots': [
                    {
                        'x': x,
                        'y': y_red2,
                        'color': 'red',
                        'threshold': 3.0,
                        'threshold_condition': 'above'
                    }
                ]
            }
        })
        
        # Graph 3: Type 1 - Another dual plot
        y_red3 = np.exp(-x/5) * np.sin(x * 3) * 2 + 1
        y_blue3 = np.exp(-x/5) * np.cos(x * 2) * 1.5 + 0.5
        row3_data.append({
            'title': f'Decaying Dual Plot - Page {page_idx+1}',
            'content': {
                'plots': [
                    {
                        'x': x,
                        'y': y_red3,
                        'color': 'red',
                        'threshold': 1.5,
                        'threshold_condition': 'above'
                    },
                    {
                        'x': x,
                        'y': y_blue3,
                        'color': 'blue',
                        'threshold': 0.3,
                        'threshold_condition': 'below'
                    }
                ]
            }
        })
        
        # Graph 4: Type 2 - Single red plot
        y_red4 = x * 0.3 + np.sin(x) * 0.5
        row3_data.append({
            'title': f'Linear Red Plot - Page {page_idx+1}',
            'content': {
                'plots': [
                    {
                        'x': x,
                        'y': y_red4,
                        'color': 'red',
                        'threshold': 2.0,
                        'threshold_condition': 'above'
                    }
                ]
            }
        })
        
        # Row 4: [graph, graph, graph, graph]
        row4_data = []
        for i in range(4):
            y = np.cos(x + i + page_idx * 0.3) * np.exp(-x/10)
            row4_data.append({
                'title': f'Simple Plot {i+1} - Page {page_idx+1}',
                'content': {
                    'plots': [
                        {
                            'x': x,
                            'y': y,
                            'color': ['blue', 'red', 'green', 'purple'][i]
                        }
                    ]
                }
            })
        
        # Add all rows to the page
        page_data = [row1_data, row2_data, row3_data, row4_data]
        dashboard.add_page(page_data)
    
    dashboard.close()