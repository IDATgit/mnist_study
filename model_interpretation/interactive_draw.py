import sys
import os
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
from scipy.ndimage import gaussian_filter

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.specific_models.StandardFullyConnected import StandardFullyConnected
from models.specific_models.StandardConvNet import StandardConvNet

def find_available_models():
    """Find all available trained models in the trainers/outputs directory."""
    models = []
    outputs_dir = os.path.join('trainers', 'outputs')
    
    if not os.path.exists(outputs_dir):
        return models
        
    for model_dir in os.listdir(outputs_dir):
        model_path = os.path.join(outputs_dir, model_dir, 'checkpoints', 'model_best.pt')
        if os.path.exists(model_path):
            # Extract model type from directory name
            model_type = 'fc' if 'fully_connected' in model_dir.lower() else 'conv'
            models.append({
                'name': model_dir,
                'path': model_path,
                'type': model_type
            })
    
    return models

class DigitDrawer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digit Recognition")
        
        # Set color scheme
        self.colors = {
            'bg': '#0f172a',           # Dark blue-gray background
            'canvas_bg': '#000000',     # Black canvas
            'accent': '#6366f1',        # Indigo accent
            'accent_hover': '#818cf8',  # Lighter indigo for hover
            'text': '#f1f5f9',          # Light gray text
            'secondary': '#10b981',     # Emerald green for high confidence
            'low_conf': '#ef4444',      # Red for low confidence
            'mid_conf': '#f59e0b',      # Amber for medium confidence
            'surface': '#1e293b',       # Lighter surface color
            'border': '#334155'         # Border color
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Find available models
        self.available_models = find_available_models()
        if not self.available_models:
            raise RuntimeError("No trained models found in trainers/outputs directory")
        
        # Load the first model by default
        self.current_model = self.available_models[0]
        self.model = self._load_model(self.current_model['path'])
        
        # Drawing settings
        self.pen_size = 20
        self.drawing_size = 280  # 10x the MNIST size for better drawing
        self.mnist_size = 28
        self.preview_size = 140  # 5x the MNIST size for better visibility
        self.last_prediction_time = 0
        self.prediction_delay = 100  # milliseconds
        
        # Shift settings
        self.shift_start = None
        
        # Configure ttk styles
        self.setup_styles()
        
        # Setup UI
        self.setup_ui()
        
        # Fix window size
        self.root.update()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())
        self.root.maxsize(self.root.winfo_width(), self.root.winfo_height())
        
    def setup_styles(self):
        """Configure ttk styles for widgets."""
        style = ttk.Style()
        
        # Configure frame style
        style.configure('Main.TFrame', background=self.colors['bg'])
        style.configure('Dark.TFrame', background=self.colors['bg'])
        
        # Configure label styles with modern font
        style.configure('Title.TLabel',
                       font=('Segoe UI', 24, 'bold'),
                       background=self.colors['bg'],
                       foreground=self.colors['text'])
                       
        style.configure('Preview.TLabel',
                       font=('Segoe UI', 12),
                       background=self.colors['bg'],
                       foreground=self.colors['text'])
                       
        style.configure('Digit.TLabel',
                       font=('Segoe UI', 14, 'bold'),
                       background=self.colors['bg'],
                       foreground=self.colors['text'])
                       
        style.configure('Percentage.TLabel',
                       font=('Segoe UI', 12),
                       background=self.colors['bg'],
                       foreground=self.colors['text'])
        
        # Configure combobox style
        style.configure('TCombobox',
                       fieldbackground=self.colors['surface'],
                       background=self.colors['surface'],
                       foreground=self.colors['text'],
                       arrowcolor=self.colors['text'],
                       selectbackground=self.colors['accent'],
                       selectforeground=self.colors['text'])
        
        # Configure progressbar styles
        style.configure('Confidence.Horizontal.TProgressbar',
                       troughcolor=self.colors['surface'],
                       background=self.colors['accent'],
                       darkcolor=self.colors['accent'],
                       lightcolor=self.colors['accent'],
                       bordercolor=self.colors['border'])
                       
        style.configure('High.Horizontal.TProgressbar',
                       troughcolor=self.colors['surface'],
                       background=self.colors['secondary'],
                       darkcolor=self.colors['secondary'],
                       lightcolor=self.colors['secondary'],
                       bordercolor=self.colors['border'])
                       
        style.configure('Mid.Horizontal.TProgressbar',
                       troughcolor=self.colors['surface'],
                       background=self.colors['mid_conf'],
                       darkcolor=self.colors['mid_conf'],
                       lightcolor=self.colors['mid_conf'],
                       bordercolor=self.colors['border'])
                       
        style.configure('Low.Horizontal.TProgressbar',
                       troughcolor=self.colors['surface'],
                       background=self.colors['low_conf'],
                       darkcolor=self.colors['low_conf'],
                       lightcolor=self.colors['low_conf'],
                       bordercolor=self.colors['border'])
        
        # Use a regular tk.Button instead of ttk.Button for better color control
        self.custom_button = {
            'font': ('Segoe UI', 12),
            'bg': self.colors['accent'],
            'fg': self.colors['text'],
            'activebackground': self.colors['accent_hover'],
            'activeforeground': self.colors['text'],
            'relief': 'flat',
            'padx': 20,
            'pady': 10,
            'border': 0
        }
        
    def _load_model(self, model_path):
        """Load the trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}")
            
        print(f"Loading model from {model_path}")
        
        # Load checkpoint first to check its structure
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['model_state_dict']
        
        # Determine model type from state dict structure
        is_fully_connected = any('model.' in key for key in state_dict.keys())
        
        # Create appropriate model
        if is_fully_connected:
            model = StandardFullyConnected()
        else:
            model = StandardConvNet()
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
        
    def on_model_change(self, event):
        """Handle model selection change."""
        selected_name = self.model_var.get()
        for model in self.available_models:
            if model['name'] == selected_name:
                self.current_model = model
                self.model = self._load_model(model['path'])
                break
        self.clear_canvas()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, style='Main.TFrame')
        main_frame.pack(padx=30, pady=30, expand=True, fill='both')
        
        # Model selection frame
        model_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        model_frame.pack(fill='x', pady=(0, 20))
        
        # Model selection label
        model_label = ttk.Label(
            model_frame,
            text="Select Model:",
            style='Preview.TLabel'
        )
        model_label.pack(side='left', padx=(0, 10))
        
        # Model selection dropdown
        self.model_var = tk.StringVar()
        self.model_var.set(self.available_models[0]['name'])
        model_names = [model['name'] for model in self.available_models]
        model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=model_names,
            state='readonly',
            width=30
        )
        model_dropdown.pack(side='left', fill='x', expand=True)
        model_dropdown.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Content frame for drawing and prediction
        content_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        content_frame.pack(fill='both', expand=True)
        
        # Left frame for drawing area and preview
        left_frame = ttk.Frame(content_frame, style='Dark.TFrame')
        left_frame.pack(side='left', padx=15, pady=10)
        
        # Drawing canvas with thicker border
        self.canvas = tk.Canvas(
            left_frame,
            width=self.drawing_size,
            height=self.drawing_size,
            bg=self.colors['canvas_bg'],
            cursor='crosshair',
            highlightthickness=3,
            highlightbackground=self.colors['accent']
        )
        self.canvas.pack(pady=(0, 20))
        
        # Preview label
        preview_label = ttk.Label(
            left_frame,
            text="28×28 Input Preview",
            style='Preview.TLabel'
        )
        preview_label.pack()
        
        # Preview canvas with subtle border
        self.preview_canvas = tk.Canvas(
            left_frame,
            width=self.preview_size,
            height=self.preview_size,
            bg=self.colors['canvas_bg'],
            highlightthickness=2,
            highlightbackground=self.colors['border']
        )
        self.preview_canvas.pack(pady=10)
        
        # Prediction frame (right side)
        pred_frame = ttk.Frame(content_frame, style='Dark.TFrame')
        pred_frame.pack(side='left', padx=15, pady=10, fill='both', expand=True)
        
        # Prediction label with pre-allocated space
        self.pred_label = ttk.Label(
            pred_frame,
            text="Draw a digit\n ",
            style='Title.TLabel',
            justify='center'
        )
        self.pred_label.pack(pady=25)
        
        # Confidence bars frame
        conf_frame = ttk.Frame(pred_frame, style='Dark.TFrame')
        conf_frame.pack(fill='both', expand=True, padx=10)
        
        # Create confidence bars
        self.conf_bars = []
        self.conf_labels = []
        for i in range(10):
            # Row frame for each digit
            row_frame = ttk.Frame(conf_frame, style='Dark.TFrame')
            row_frame.pack(fill='x', pady=3)
            
            # Digit label
            digit_label = ttk.Label(row_frame, text=str(i), style='Digit.TLabel')
            digit_label.pack(side='left', padx=15)
            
            # Progress bar
            bar = ttk.Progressbar(
                row_frame,
                length=200,
                mode='determinate',
                style='Confidence.Horizontal.TProgressbar'
            )
            bar.pack(side='left', padx=8, fill='x', expand=True)
            
            # Percentage label
            conf_label = ttk.Label(row_frame, text="0.0%", style='Percentage.TLabel', width=8)
            conf_label.pack(side='left', padx=15)
            
            self.conf_bars.append(bar)
            self.conf_labels.append(conf_label)
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        btn_frame.pack(fill='x', pady=20)
        
        # Clear button with modern style using tk.Button instead of ttk.Button
        clear_btn = tk.Button(
            btn_frame,
            text="Clear Canvas",
            command=self.clear_canvas,
            **self.custom_button
        )
        clear_btn.pack(side='left', padx=5)
        
        # Setup drawing events
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<Button-1>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        
        # Setup shift events
        self.canvas.bind('<Button-3>', self.start_shift)
        self.canvas.bind('<B3-Motion>', self.shift_image)
        self.canvas.bind('<ButtonRelease-3>', self.end_shift)
        
        # Create PIL image for drawing
        self.image = Image.new('L', (self.drawing_size, self.drawing_size), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
    def start_shift(self, event):
        """Start the shift operation."""
        self.shift_start = (event.x, event.y)
        self.canvas.configure(cursor='fleur')
            
    def shift_image(self, event):
        """Shift the image based on mouse movement."""
        if self.shift_start:
            # Calculate shift amount
            dx = event.x - self.shift_start[0]
            dy = event.y - self.shift_start[1]
            
            # Create new image with black background
            new_image = Image.new('L', (self.drawing_size, self.drawing_size), 'black')
            
            # Paste the current image with offset
            new_image.paste(self.image, (dx, dy))
            
            # Update the image
            self.image = new_image
            self.draw = ImageDraw.Draw(self.image)
            
            # Update canvas
            self.canvas.delete('all')
            self.canvas_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor='nw', image=self.canvas_image)
            
            # Update preview and predict
            self.update_preview()
            self.predict()
            
            # Update shift start position
            self.shift_start = (event.x, event.y)
            
    def end_shift(self, event):
        """End the shift operation."""
        self.shift_start = None
        self.canvas.configure(cursor='crosshair')
            
    def draw(self, event):
        """Handle drawing on canvas."""
        x, y = event.x, event.y
        x1, y1 = x - self.pen_size, y - self.pen_size
        x2, y2 = x + self.pen_size, y + self.pen_size
        
        # Draw on canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        
        # Draw on PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill='white')
        
        # Update preview
        self.update_preview()
        
        # Schedule prediction with delay
        current_time = self.root.tk.call('clock', 'milliseconds')
        if current_time - self.last_prediction_time >= self.prediction_delay:
            self.predict()
            self.last_prediction_time = current_time
            
    def on_release(self, event):
        """Handle mouse release event."""
        # Force a prediction when releasing the mouse button
        self.predict()
        
    def clear_canvas(self):
        """Clear the canvas and reset predictions."""
        self.canvas.delete('all')
        self.preview_canvas.delete('all')
        self.image = Image.new('L', (self.drawing_size, self.drawing_size), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.pred_label.config(text="Draw a digit\n ")  # Keep the extra line
        
        # Reset confidence bars
        for bar, label in zip(self.conf_bars, self.conf_labels):
            bar['value'] = 0
            label.config(text="0.0%")
        
    def predict(self):
        """Predict the drawn digit."""
        # Convert to MNIST size (28x28)
        img_array = np.array(self.image)
        img_small = Image.fromarray(img_array).resize((self.mnist_size, self.mnist_size), 
                                                     Image.Resampling.LANCZOS)
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(np.array(img_small)).reshape(1, -1) / 255.0
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().numpy()
            prediction = probabilities.argmax()
            
        # Update prediction label
        confidence = probabilities[prediction] * 100
        self.pred_label.config(
            text=f"Prediction: {prediction}\nConfidence: {confidence:.1f}%"
        )
        
        # Update confidence bars with color coding
        max_prob = probabilities.max()
        for i, (prob, bar, label) in enumerate(zip(probabilities, self.conf_bars, self.conf_labels)):
            bar['value'] = prob * 100
            label.config(text=f"{prob*100:.1f}%")
            
            # Color the bar based on probability
            if prob == max_prob:
                bar.configure(style='High.Horizontal.TProgressbar')
            elif prob > 0.2:
                bar.configure(style='Mid.Horizontal.TProgressbar')
            else:
                bar.configure(style='Low.Horizontal.TProgressbar')
        
    def update_preview(self):
        """Update the preview of the 28x28 MNIST input."""
        # Get the 28x28 version
        img_array = np.array(self.image)
        img_small = Image.fromarray(img_array).resize((self.mnist_size, self.mnist_size), 
                                                     Image.Resampling.LANCZOS)
        
        # Scale it up for better visibility
        img_preview = img_small.resize((self.preview_size, self.preview_size), 
                                     Image.Resampling.NEAREST)
        
        # Convert to PhotoImage for Tkinter
        self.preview_image = ImageTk.PhotoImage(img_preview)
        
        # Update preview canvas
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(0, 0, anchor='nw', image=self.preview_image)
        
    def run(self):
        """Start the application."""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = DigitDrawer()
        app.run()
    except FileNotFoundError as e:
        print(e)
        print("Please train a model first using basic_trainer.py") 