import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import math

class ImageDetectiveGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Detective Game")
        self.root.geometry("1200x800")
        
        # Game state
        self.current_level = 1
        self.original_image = None
        self.distorted_image = None
        self.current_image = None
        self.score = 0
        self.max_levels = 5
        self.game_started = False

        # Preload clue images
        self.clue_images = []
        self.preload_clue_images()
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top panel - Game info
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text="Level:").pack(side=tk.LEFT)
        self.level_label = ttk.Label(info_frame, text="1", font=("Arial", 12, "bold"))
        self.level_label.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(info_frame, text="Score:").pack(side=tk.LEFT)
        self.score_label = ttk.Label(info_frame, text="0", font=("Arial", 12, "bold"))
        self.score_label.pack(side=tk.LEFT, padx=(5, 20))
        
        self.next_level_btn = ttk.Button(info_frame, text="Start Game", command=self.start_game_or_next_level)
        self.next_level_btn.pack(side=tk.RIGHT)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Tools
        tools_frame = ttk.LabelFrame(content_frame, text="Processing Tools", padding=10)
        tools_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Noise Reduction Tools
        noise_frame = ttk.LabelFrame(tools_frame, text="Noise Reduction")
        noise_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(noise_frame, text="Gaussian Filter", 
                  command=self.apply_gaussian_filter).pack(fill=tk.X, pady=2)
        self.gaussian_slider = ttk.Scale(noise_frame, from_=1, to=15, orient=tk.HORIZONTAL)
        self.gaussian_slider.set(5)
        self.gaussian_slider.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, text="Gaussian Kernel Size").pack()

        ttk.Button(noise_frame, text="Median Filter", 
                  command=self.apply_median_filter).pack(fill=tk.X, pady=2)
        self.median_slider = ttk.Scale(noise_frame, from_=3, to=11, orient=tk.HORIZONTAL)
        self.median_slider.set(5)
        self.median_slider.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, text="Median Kernel Size").pack()

        ttk.Button(noise_frame, text="Bilateral Filter", 
                  command=self.apply_bilateral_filter).pack(fill=tk.X, pady=2)
        
        # Sharpening Tools
        sharp_frame = ttk.LabelFrame(tools_frame, text="Sharpening")
        sharp_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(sharp_frame, text="Unsharp Masking", 
                  command=self.apply_unsharp_mask).pack(fill=tk.X, pady=2)
        ttk.Button(sharp_frame, text="Laplacian Sharpen", 
                  command=self.apply_laplacian_sharpen).pack(fill=tk.X, pady=2)
        
        # Enhancement Tools
        enhance_frame = ttk.LabelFrame(tools_frame, text="Enhancement")
        enhance_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(enhance_frame, text="Hist Equalization", 
                  command=self.apply_histogram_eq).pack(fill=tk.X, pady=2)
        ttk.Button(enhance_frame, text="Gamma Correction", 
                  command=self.apply_gamma_correction).pack(fill=tk.X, pady=2)
        self.gamma_slider = tk.Scale(enhance_frame, from_=0.5, to=2.5, orient=tk.HORIZONTAL, resolution=0.1)
        self.gamma_slider.set(1.2)
        self.gamma_slider.pack(fill=tk.X, pady=2)
        ttk.Label(enhance_frame, text="Gamma Value").pack()
        
        # Control buttons
        ttk.Button(tools_frame, text="Reset Image", 
                  command=self.reset_image).pack(fill=tk.X, pady=10)
        ttk.Button(tools_frame, text="Check Solution", 
                  command=self.check_solution).pack(fill=tk.X, pady=2)
        
        # Right panel - Image display
        image_frame = ttk.Frame(content_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image canvas
        self.canvas = tk.Canvas(image_frame, bg="white", width=600, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Clue display
        self.clue_label = ttk.Label(image_frame, text="Find the hidden clue!", 
                                   font=("Arial", 14), foreground="blue")
        self.clue_label.pack(pady=10)
        
    def preload_clue_images(self):
        # Example: Load images from files
        image_paths = [
            "e:/Image Processing Lab/Project/Clue-1.png",
            "e:/Image Processing Lab/Project/Clue-2.png",
            "e:/Image Processing Lab/Project/Clue-3.png",
            "e:/Image Processing Lab/Project/Clue-4.png",
            "e:/Image Processing Lab/Project/Clue-5.jpg",
            "e:/Image Processing Lab/Project/Clue-6.jpg"
        ]
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                # Resize if too large
                h, w = img.shape[:2]
                if h > 400 or w > 400:
                    scale = min(400/h, 400/w)
                    new_h, new_w = int(h*scale), int(w*scale)
                    img = cv2.resize(img, (new_w, new_h))
                self.clue_images.append(img)
    
    def set_level_image(self):
        # Set the image for the current level
        if self.current_level <= len(self.clue_images):
            self.original_image = self.clue_images[self.current_level - 1].copy()
            self.apply_distortion()

    def start_game_or_next_level(self):
        if not self.game_started:
            self.game_started = True
            self.next_level_btn.config(text="Next Level")
            self.set_level_image()
        else:
            self.start_new_level()
    
    def apply_distortion(self):
        if self.original_image is None:
            return

        img = self.original_image.copy()

        if self.current_level == 1:
            # Moderate Gaussian blur
            img = cv2.GaussianBlur(img, (9, 9), 3)
        
        elif self.current_level == 2:
            # Mild salt-and-pepper noise
            s_vs_p = 0.5
            amount = 0.02  # 2% pixels
            out = img.copy()
            num_salt = np.ceil(amount * img.size * s_vs_p / img.shape[2])
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
            out[coords[0], coords[1], :] = 255
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p) / img.shape[2])
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
            out[coords[0], coords[1], :] = 0
            img = out

        elif self.current_level == 3:
            # Moderate contrast reduction
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(img)
            y = cv2.normalize(y, None, 100, 150, cv2.NORM_MINMAX)
            img = cv2.merge([y, cr, cb])
            img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

        elif self.current_level == 4:
            # Moderate gamma darkening
            gamma = 1.6
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)

        elif self.current_level == 5:
            # Mild Gaussian noise
            noise = np.random.normal(0, 10, img.shape).astype(np.float32)
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)

        else:
            img = cv2.GaussianBlur(img, (5, 5), 1)
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)

        self.distorted_image = img
        self.current_image = img.copy()
        self.display_image()
    
    def display_image(self):
        if self.current_image is None:
            return
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize for display
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            img_pil.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and display
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                               image=self.photo, anchor=tk.CENTER)
    
    # Image Processing Methods
    
    def apply_gaussian_filter(self):
        if self.current_image is not None:
            ksize = int(self.gaussian_slider.get())
            if ksize % 2 == 0: ksize += 1  # Kernel size must be odd
            self.current_image = cv2.GaussianBlur(self.current_image, (ksize, ksize), 1)
            self.display_image()
    
    def apply_median_filter(self):
        if self.current_image is not None:
            ksize = int(self.median_slider.get())
            if ksize % 2 == 0: ksize += 1
            self.current_image = cv2.medianBlur(self.current_image, ksize)
            self.display_image()
    
    def apply_bilateral_filter(self):
        if self.current_image is not None:
            self.current_image = cv2.bilateralFilter(self.current_image, 9, 75, 75)
            self.display_image()
    
    def apply_unsharp_mask(self):
        if self.current_image is not None:
            gaussian = cv2.GaussianBlur(self.current_image, (0, 0), 2.0)
            self.current_image = cv2.addWeighted(self.current_image, 1.5, gaussian, -0.5, 0)
            self.display_image()
    
    def apply_laplacian_sharpen(self):
        if self.current_image is not None:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            self.current_image = cv2.filter2D(self.current_image, -1, kernel)
            self.display_image()
    
    def apply_histogram_eq(self):
        if self.current_image is not None:
            # Apply to each channel
            channels = cv2.split(self.current_image)
            eq_channels = [cv2.equalizeHist(ch) for ch in channels]
            self.current_image = cv2.merge(eq_channels)
            self.display_image()
    
    def apply_gamma_correction(self):
        if self.current_image is not None:
            gamma = float(self.gamma_slider.get())
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
            self.current_image = cv2.LUT(self.current_image, table)
            self.display_image()
    
    def reset_image(self):
        if self.distorted_image is not None:
            self.current_image = self.distorted_image.copy()
            self.display_image()
    
    def check_solution(self):
        if self.current_image is None or self.original_image is None:
            return
        
        # Calculate PSNR as quality metric
        mse = np.mean((self.current_image.astype(float) - self.original_image.astype(float)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * math.log10(255.0 / math.sqrt(mse))
        
        # Score calculation
        if psnr > 25:
            points = 100
            message = "Excellent! Perfect restoration!"
        elif psnr > 20:
            points = 80
            message = "Good job! Very close!"
        elif psnr > 15:
            points = 60
            message = "Not bad, but could be better."
        else:
            points = 20
            message = "Keep trying! Apply more processing."
        
        self.score += points
        self.score_label.config(text=str(self.score))
        
        messagebox.showinfo("Solution Check", 
                          f"{message}\nPSNR: {psnr:.2f} dB\nPoints: {points}")
    
    def start_new_level(self):
        if self.current_level < self.max_levels:
            self.current_level += 1
            self.level_label.config(text=str(self.current_level))
            self.set_level_image()
        else:
            messagebox.showinfo("Game Complete", 
                              f"Congratulations! Final Score: {self.score}")
            self.current_level=1
            self.score=0
            self.level_label.config(text=str(self.current_level))
            self.score_label.config(text=str(self.score))
            self.current_image=None
            self.distorted_image=None
            self.original_image=None
            self.canvas.delete("all")
            self.next_level_btn.config(text="Restart Game")
            self.game_started = False 

if __name__ == "__main__":
    root = tk.Tk()
    game = ImageDetectiveGame(root)
    root.mainloop()