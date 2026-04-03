import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import cv2
from PIL import Image, ImageTk
import math
import random

class ClueHuntingGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Detective Clue Hunter Game")
        self.root.geometry("1200x800")
        
        self.current_level = 1
        self.original_image = None
        self.distorted_image = None
        self.current_image = None
        self.score = 0
        self.max_levels = 3
        self.game_started = False
        
        self.image_history = []
        self.max_history = 10 
        
        self.gaussian_kernel_size = tk.IntVar(value=5)
        self.gaussian_sigma = tk.DoubleVar(value=1.5)
        self.median_kernel_size = tk.IntVar(value=5)
        self.bilateral_d = tk.IntVar(value=9)
        self.bilateral_sigma_color = tk.IntVar(value=75)
        self.bilateral_sigma_space = tk.IntVar(value=75)
        self.gamma_value = tk.DoubleVar(value=2.2)
        self.contrast_alpha = tk.DoubleVar(value=1.5)
        self.contrast_beta = tk.IntVar(value=30)
        
        # For Level 3 encoding keys
        self.encoded_images = {}  # Store encoded images with their keys
        
        self.level_clues = {
            1: {
                "question": "Find out the ID no",
                "answer": "735918642A",  
                "hint": "Hmmm...What can be the opposite of bluring? Maybe its sharpening."
            },
            2: {
                "question": "Find out the codes from the image.",
                "answer": "1234,5678",
                "hint": "This much noise!!!Median Filter works well for salt pepper"
            },
            3: {
                "question": "Decode the image using the correct keys to reveal the meeting date!",
                "answer": "Tuesday",
                "hint": "The image is encoded in frequency domain. You need TWO keys to decode it!Try the ones found in the previous image",
                "keys": {"phase": 1234, "perm": 5678} 
            }
        }

        self.clue_images = []
        self.preload_clue_images()
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text="Level:").pack(side=tk.LEFT)
        self.level_label = ttk.Label(info_frame, text="1", font=("Arial", 12, "bold"))
        self.level_label.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(info_frame, text="Score:").pack(side=tk.LEFT)
        self.score_label = ttk.Label(info_frame, text="0", font=("Arial", 12, "bold"))
        self.score_label.pack(side=tk.LEFT, padx=(5, 20))
        
        self.start_btn = ttk.Button(info_frame, text="Start Game", command=self.start_game)
        self.start_btn.pack(side=tk.RIGHT)
        
        question_frame = ttk.LabelFrame(main_frame, text="Your Mission", padding=10)
        question_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.question_label = ttk.Label(question_frame, text="Click Start Game to begin your detective mission!", 
                                       font=("Arial", 12), wraplength=800)
        self.question_label.pack()
        
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        tools_container = ttk.LabelFrame(content_frame, text="Image Processing Tools", padding=2)
        tools_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        tools_canvas = tk.Canvas(tools_container, width=150, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tools_container, orient="vertical", command=tools_canvas.yview)
        tools_frame = ttk.Frame(tools_canvas)
        
        tools_frame.bind(
            "<Configure>",
            lambda e: tools_canvas.configure(scrollregion=tools_canvas.bbox("all"))
        )
        
        tools_canvas.create_window((0, 0), window=tools_frame, anchor="nw")
        tools_canvas.configure(yscrollcommand=scrollbar.set)
        
        tools_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def _on_mousewheel(event):
            tools_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        tools_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        noise_frame = ttk.LabelFrame(tools_frame, text="Noise Reduction", padding=5)
        noise_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(noise_frame, text="Gaussian Filter", 
                  command=self.apply_gaussian_filter).pack(fill=tk.X, pady=2)
        
        ttk.Label(noise_frame, text="Kernel Size:", font=("Arial", 8)).pack(anchor=tk.W, padx=5, pady=(5,0))
        gaussian_k_scale = ttk.Scale(noise_frame, from_=3, to=15, orient=tk.HORIZONTAL,
                                     variable=self.gaussian_kernel_size, command=self._round_odd)
        gaussian_k_scale.pack(fill=tk.X, padx=5)
        self.gaussian_k_label = ttk.Label(noise_frame, text="5", font=("Arial", 8))
        self.gaussian_k_label.pack(anchor=tk.W, padx=5)
        
        ttk.Label(noise_frame, text="Sigma:", font=("Arial", 8)).pack(anchor=tk.W, padx=5, pady=(3,0))
        gaussian_s_scale = ttk.Scale(noise_frame, from_=0.5, to=5.0, orient=tk.HORIZONTAL,
                                     variable=self.gaussian_sigma, command=self._update_gaussian_sigma_label)
        gaussian_s_scale.pack(fill=tk.X, padx=5)
        self.gaussian_s_label = ttk.Label(noise_frame, text="1.5", font=("Arial", 8))
        self.gaussian_s_label.pack(anchor=tk.W, padx=5, pady=(0,5))
        
        ttk.Separator(noise_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        ttk.Button(noise_frame, text="Median Filter", 
                  command=self.apply_median_filter).pack(fill=tk.X, pady=2)
        
        ttk.Label(noise_frame, text="Kernel Size:", font=("Arial", 8)).pack(anchor=tk.W, padx=5, pady=(5,0))
        median_scale = ttk.Scale(noise_frame, from_=3, to=11, orient=tk.HORIZONTAL,
                                variable=self.median_kernel_size, command=self._round_odd_median)
        median_scale.pack(fill=tk.X, padx=5)
        self.median_label = ttk.Label(noise_frame, text="5", font=("Arial", 8))
        self.median_label.pack(anchor=tk.W, padx=5, pady=(0,5))
        
        ttk.Separator(noise_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        ttk.Button(noise_frame, text="Bilateral Filter", 
                  command=self.apply_bilateral_filter).pack(fill=tk.X, pady=2)
        
        sharp_frame = ttk.LabelFrame(tools_frame, text="Sharpening", padding=5)
        sharp_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(sharp_frame, text="Unsharp Masking", 
                  command=self.apply_unsharp_mask).pack(fill=tk.X, pady=2)
        ttk.Button(sharp_frame, text="Laplacian Sharpen", 
                  command=self.apply_laplacian_sharpen).pack(fill=tk.X, pady=2)
        
        enhance_frame = ttk.LabelFrame(tools_frame, text="Enhancement", padding=5)
        enhance_frame.pack(fill=tk.X, pady=(0, 10))
    
        ttk.Button(enhance_frame, text="Gamma Correction", 
                  command=self.apply_gamma_correction).pack(fill=tk.X, pady=2)
        
        ttk.Label(enhance_frame, text="Gamma:", font=("Arial", 8)).pack(anchor=tk.W, padx=5, pady=(5,0))
        gamma_scale = ttk.Scale(enhance_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL,
                               variable=self.gamma_value, command=self._update_gamma_label)
        gamma_scale.pack(fill=tk.X, padx=5)
        self.gamma_label = ttk.Label(enhance_frame, text="2.2", font=("Arial", 8))
        self.gamma_label.pack(anchor=tk.W, padx=5, pady=(0,5))
        
        ttk.Separator(enhance_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        ttk.Button(enhance_frame, text="Contrast Stretch", 
                  command=self.apply_contrast_stretch).pack(fill=tk.X, pady=2)
        
        ttk.Label(enhance_frame, text="Alpha (Contrast):", font=("Arial", 8)).pack(anchor=tk.W, padx=5, pady=(5,0))
        alpha_scale = ttk.Scale(enhance_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL,
                               variable=self.contrast_alpha, command=self._update_alpha_label)
        alpha_scale.pack(fill=tk.X, padx=5)
        self.alpha_label = ttk.Label(enhance_frame, text="1.5", font=("Arial", 8))
        self.alpha_label.pack(anchor=tk.W, padx=5)
        
        ttk.Label(enhance_frame, text="Beta (Brightness):", font=("Arial", 8)).pack(anchor=tk.W, padx=5, pady=(3,0))
        beta_scale = ttk.Scale(enhance_frame, from_=-50, to=100, orient=tk.HORIZONTAL,
                              variable=self.contrast_beta, command=self._update_beta_label)
        beta_scale.pack(fill=tk.X, padx=5)
        self.beta_label = ttk.Label(enhance_frame, text="30", font=("Arial", 8))
        self.beta_label.pack(anchor=tk.W, padx=5, pady=(0,5))
        
        freq_frame = ttk.LabelFrame(tools_frame, text="Frequency Decoding", padding=5)
        freq_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(freq_frame, text="Decode Image with Keys", 
                  command=self.decode_frequency_image).pack(fill=tk.X, pady=2)
        ttk.Label(freq_frame, text="Enter phase & perm keys", font=("Arial", 7), 
                 foreground="gray").pack(anchor=tk.W, padx=5)
        
        # Periodic Noise Removal Frame
        periodic_frame = ttk.LabelFrame(tools_frame, text="Periodic Noise Removal", padding=5)
        periodic_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(periodic_frame, text="Auto Notch Filter", 
                  command=self.apply_notch_filter).pack(fill=tk.X, pady=2)
        ttk.Label(periodic_frame, text="Removes periodic patterns", font=("Arial", 7), 
                 foreground="gray").pack(anchor=tk.W, padx=5)
        
        control_frame = ttk.LabelFrame(tools_frame, text="Controls", padding=5)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Undo Last Action", 
                  command=self.undo_last_action).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Reset Image", 
                  command=self.reset_image).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Get Hint", 
                  command=self.show_hint).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Submit Answer", 
                  command=self.submit_answer).pack(fill=tk.X, pady=2)
        
        image_frame = ttk.Frame(content_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(image_frame, bg="white", width=600, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
    def preload_clue_images(self):
        image_paths = [
            "e:/Image Processing Lab/Project/Clue-1.png",
            "e:/Image Processing Lab/Project/Clue-2e.png", 
            "e:/Image Processing Lab/Project/Clue-3.png",
        ]
        
        for idx, path in enumerate(image_paths):
            if idx == 2:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape
                    if h > 500 or w > 500:
                        scale = min(500/h, 500/w)
                        new_h, new_w = int(h*scale), int(w*scale)
                        img = cv2.resize(img, (new_w, new_h))
                    self.clue_images.append(img)
            else:
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    if h > 500 or w > 500:
                        scale = min(500/h, 500/w)
                        new_h, new_w = int(h*scale), int(w*scale)
                        img = cv2.resize(img, (new_w, new_h))
                    self.clue_images.append(img)
                
        
    def start_game(self):
        if not self.game_started:
            self.game_started=True
            self.current_level=1
            self.score=0
            self.start_btn.config(text="Restart Game")
            self.level_label.config(text="1")
            self.score_label.config(text="0")
            self.image_history.clear()  # Clear history on new game
            self.encoded_images.clear()  # Clear encoded images
            self.load_level()
        else:
            # Restart the game
            self.end_game()
            self.start_game()
    
    def end_game(self):
        """End the current game and reset to initial state"""
        self.game_started = False
        self.current_level = 1
        self.score = 0
        self.current_image = None
        self.original_image = None
        self.distorted_image = None
        self.image_history.clear()
        self.encoded_images.clear()
        self.start_btn.config(text="Start Game")
        self.level_label.config(text="1")
        self.score_label.config(text="0")
        self.canvas.delete("all")
        self.question_label.config(text="Click Start Game to begin your detective mission!")
    
    def load_level(self):
        if self.current_level<=len(self.clue_images):
            self.original_image=self.clue_images[min(self.current_level-1, len(self.clue_images)-1)].copy()
            self.apply_hiding_distortion()
            clue_info=self.level_clues[self.current_level]
            self.question_label.config(text=f"Level {self.current_level}: {clue_info['question']}")
    
    def apply_hiding_distortion(self):
        if self.original_image is None:
            return
            
        img=self.original_image.copy()
        
        if self.current_level==1:
            img=cv2.GaussianBlur(img,(11, 11),1.8)
            img=cv2.convertScaleAbs(img,alpha=0.4,beta=10)
        
            
        elif self.current_level==2:
            img=cv2.convertScaleAbs(img, alpha=0.4, beta=-30)
    
            s_vs_p = 0.5
            amount = 0.04
            noisy = img.copy()
    
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
            noisy[coords[0], coords[1]] = 255
    
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
            noisy[coords[0], coords[1]] = 0
    
            img = noisy
            
        elif self.current_level==3:
            img = self.encode_image_frequency(img, key_phase=1234, key_perm=5678)
            
            self.encoded_images[3] = {
                'encoded_gray': img.copy(),
                'keys': {'phase': 1234, 'perm': 5678}
            }
           
            
        
        self.distorted_image=img
        self.current_image=img.copy()
        
        self.image_history=[]
        self.save_to_history()
        
        self.display_image()
        
    def display_image(self):
        if self.current_image is None:
            return
        
        if len(self.current_image.shape) == 2:
            img_pil = Image.fromarray(self.current_image)
        else:
            img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width>1 and canvas_height>1:
            img_pil.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
        
        self.photo=ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                               image=self.photo, anchor=tk.CENTER)
    
    def save_to_history(self):
        if self.current_image is not None:
            self.image_history.append(self.current_image.copy())
            
            if len(self.image_history)>self.max_history:
                self.image_history.pop(0)
    
    def undo_last_action(self):
        if len(self.image_history)<=1:
            messagebox.showinfo("Undo", "No more actions to undo!")
            return
        
        self.image_history.pop()
        self.current_image = self.image_history[-1].copy()
        self.display_image()
    
    def reset_image(self):
        if self.distorted_image is not None:
            self.current_image = self.distorted_image.copy()
            self.image_history = [self.current_image.copy()]
            self.display_image()
    
    def _round_odd(self, val):
        value = int(float(val))
        if value % 2 == 0:
            value += 1
        self.gaussian_kernel_size.set(value)
        self.gaussian_k_label.config(text=str(value))
    
    def _round_odd_median(self, val):
        value = int(float(val))
        if value % 2 == 0:
            value += 1
        self.median_kernel_size.set(value)
        self.median_label.config(text=str(value))
    
    def _update_gaussian_sigma_label(self, val):
        self.gaussian_s_label.config(text=f"{float(val):.2f}")
    
    def _update_gamma_label(self, val):
        self.gamma_label.config(text=f"{float(val):.2f}")
    
    def _update_alpha_label(self, val):
        self.alpha_label.config(text=f"{float(val):.2f}")
    
    def _update_beta_label(self, val):
        self.beta_label.config(text=str(int(float(val))))
    
    def apply_gaussian_filter(self):
        if self.current_image is not None:
            self.save_to_history()
            k_size = self.gaussian_kernel_size.get()
            sigma = self.gaussian_sigma.get()
            self.current_image=cv2.GaussianBlur(self.current_image, (k_size, k_size), sigma)
            self.display_image()
    
    def apply_median_filter(self):
        if self.current_image is not None:
            self.save_to_history()
            k_size = self.median_kernel_size.get()
            self.current_image = cv2.medianBlur(self.current_image, k_size)
            self.display_image()
    
    def apply_bilateral_filter(self):
        if self.current_image is not None:
            self.save_to_history()
            self.current_image = cv2.bilateralFilter(self.current_image, 9, 75, 75)
            self.display_image()
    
    def apply_unsharp_mask(self):
        if self.current_image is not None:
            self.save_to_history()
            gaussian = cv2.GaussianBlur(self.current_image, (0, 0), 2.0)
            self.current_image = cv2.addWeighted(self.current_image, 1.5, gaussian, -0.5, 0)
            self.display_image()
    
    def apply_laplacian_sharpen(self):
        if self.current_image is not None:
            self.save_to_history()
            blurred=cv2.GaussianBlur(self.current_image, (3, 3), 1.0)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            self.current_image = cv2.filter2D(blurred, -1, kernel)
            self.display_image()
    
    
    def apply_gamma_correction(self):
        if self.current_image is not None:
            self.save_to_history()
            gamma = self.gamma_value.get()
            inv_gamma = 1.0 / gamma
            table = np.array([((i/255.0)**inv_gamma)*255 
                            for i in np.arange(0, 256)]).astype("uint8")
            self.current_image = cv2.LUT(self.current_image, table)
            self.display_image()
    
    def apply_contrast_stretch(self):
        if self.current_image is not None:
            self.save_to_history()
            alpha = self.contrast_alpha.get()
            beta = self.contrast_beta.get()
            self.current_image = cv2.convertScaleAbs(self.current_image, alpha=alpha, beta=beta)
            self.display_image()
    
    def encode_image_frequency(self, img, key_phase, key_perm):
        M, N = img.shape
        
        F = np.fft.fft2(img)
        F_shifted = np.fft.fftshift(F)

        magnitude = np.abs(F_shifted)
        phase = np.angle(F_shifted)

        np.random.seed(key_phase)
        phase_mask = np.random.uniform(0, 2*np.pi, (M, N))

        encoded_phase = (phase + phase_mask) % (2*np.pi)

        F_phase_encoded = magnitude * np.exp(1j * encoded_phase)

        coeffs = F_phase_encoded.flatten()

        np.random.seed(key_perm) 
        perm = np.random.permutation(len(coeffs))
        scrambled_coeffs = coeffs[perm]

        F_encoded = scrambled_coeffs.reshape(M, N)

        F_encoded_shifted = np.fft.ifftshift(F_encoded)
        img_encoded = np.fft.ifft2(F_encoded_shifted).real
        img_encoded = np.clip(img_encoded, 0, 255).astype(np.uint8)
        
        return img_encoded
    
    def decode_image_frequency(self, img_encoded, key_phase, key_perm):
        M, N = img_encoded.shape
        
        F = np.fft.fft2(img_encoded)
        F_shifted = np.fft.fftshift(F)
        
        coeffs = F_shifted.flatten()
        np.random.seed(key_perm)
        perm = np.random.permutation(len(coeffs))
        inverse_perm = np.argsort(perm)
        descrambled_coeffs = coeffs[inverse_perm].reshape(M, N)
        
        magnitude = np.abs(descrambled_coeffs)
        phase = np.angle(descrambled_coeffs)
        
        np.random.seed(key_phase)
        phase_mask = np.random.uniform(0, 2*np.pi, (M, N))
        decoded_phase = (phase - phase_mask) % (2*np.pi)
        
        F_decoded = magnitude * np.exp(1j * decoded_phase)
        F_decoded_shifted = np.fft.ifftshift(F_decoded)
        img_decoded = np.fft.ifft2(F_decoded_shifted).real
        img_decoded = np.clip(img_decoded, 0, 255).astype(np.uint8)
        
        return img_decoded
    
    def decode_frequency_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to decode!")
            return
        
        if self.current_level == 3 and 3 in self.encoded_images:
            img_encoded = self.encoded_images[3]['encoded_gray']
        else:
            messagebox.showwarning("Error", "No encoded image found!")
            return
        
        key_phase = simpledialog.askinteger("Phase Key", 
                                           "Enter the phase encoding key (4 digits):",
                                           minvalue=0, maxvalue=9999)
        if key_phase is None:
            return
        
        key_perm = simpledialog.askinteger("Permutation Key", 
                                          "Enter the permutation key (4 digits):",
                                          minvalue=0, maxvalue=9999)
        if key_perm is None:
            return
        
        try:
            self.save_to_history()
            decoded_img = self.decode_image_frequency(self.current_image, key_phase, key_perm)
            self.current_image = decoded_img
            self.display_image()
            
            if self.current_level == 3:
                correct_keys = self.level_clues[3]["keys"]
                if key_phase == correct_keys["phase"] and key_perm == correct_keys["perm"]:
                    messagebox.showinfo("Success!", "Correct keys! The image has been decoded successfully!")
                else:
                    messagebox.showwarning("Incorrect Keys", 
                                         f"Those keys didn't work correctly. Keep trying!\n"
                                         f"(Hint: Both keys are 4-digit numbers)")
        except Exception as e:
            messagebox.showerror("Error", f"Decoding failed: {str(e)}")
    
    def detect_notch_points(self, magnitude_spectrum, threshold_percentile=99.5, min_distance=10):
        """
        Automatically detect notch points (periodic noise) in magnitude spectrum
        
        Parameters:
        - magnitude_spectrum: FFT magnitude spectrum
        - threshold_percentile: Percentile threshold to detect peaks (higher = fewer peaks)
        - min_distance: Minimum distance between detected peaks
        
        Returns:
        - notch_pairs: List of detected notch coordinates (excluding DC component)
        """
        center_u, center_v = magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2
        
        # Apply log transform for better visualization
        mag_log = np.log(magnitude_spectrum + 1)
        
        # Set DC component to 0 to avoid detecting it
        mag_log[center_u, center_v] = 0
        
        # Find threshold based on percentile
        threshold = np.percentile(mag_log, threshold_percentile)
        
        # Find peaks above threshold
        peaks = mag_log > threshold
        
        # Get coordinates of peaks
        peak_coords = np.argwhere(peaks)
        
        # Filter out peaks too close to center (DC component)
        notch_pairs = []
        for coord in peak_coords:
            u, v = coord[0], coord[1]
            # Skip if too close to center
            dist_from_center = np.sqrt((u - center_u)**2 + (v - center_v)**2)
            if dist_from_center > min_distance:
                notch_pairs.append((u, v))
        
        # Remove duplicates that are too close to each other
        filtered_notches = []
        for i, notch in enumerate(notch_pairs):
            too_close = False
            for existing in filtered_notches:
                dist = np.sqrt((notch[0] - existing[0])**2 + (notch[1] - existing[1])**2)
                if dist < min_distance:
                    too_close = True
                    break
            if not too_close:
                filtered_notches.append(notch)
        
        return filtered_notches

    def calc_dist(self, u, v, notch):
        """Calculate distance from point (u,v) to notch point"""
        return np.sqrt((u - notch[0])**2 + (v - notch[1])**2)

    def calc_HNR(self, shape, notch, anti_notch, D0k, n):
        """Calculate notch reject filter for a notch pair"""
        M, N = shape
        u, v = np.meshgrid(np.arange(M), np.arange(N))

        Dk = self.calc_dist(u, v, notch)
        Dk_neg = self.calc_dist(u, v, anti_notch)

        Dk = np.where(Dk == 0, 1e-10, Dk)
        Dk_neg = np.where(Dk_neg == 0, 1e-10, Dk_neg)

        H1 = 1.0 / (1.0 + (D0k / Dk)**(2*n))
        H2 = 1.0 / (1.0 + (D0k / Dk_neg)**(2*n))

        return H1 * H2

    def gen_mask(self, shape, notch_pairs, anti_notch_pairs, D0k, n):
        """Generate combined notch reject filter mask"""
        mask = np.ones(shape, dtype=np.float64)

        for notch, anti_notch in zip(notch_pairs, anti_notch_pairs):
            HNR = self.calc_HNR(shape, notch, anti_notch, D0k, n)
            mask *= HNR

        return mask

    def apply_notch_filter(self):
        """Apply automatic notch filter to remove periodic noise"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded!")
            return
        
        self.save_to_history()
        
        try:
            # Handle both grayscale and color images
            if len(self.current_image.shape) == 2:
                # Grayscale image
                filtered_img = self._apply_notch_to_channel(self.current_image)
                self.current_image = filtered_img
            else:
                # Color image - process each channel
                b, g, r = cv2.split(self.current_image)
                filtered_channels = []
                
                for channel in [b, g, r]:
                    filtered_channel = self._apply_notch_to_channel(channel)
                    filtered_channels.append(filtered_channel)
                
                self.current_image = cv2.merge(filtered_channels)
            
            self.display_image()
            messagebox.showinfo("Success", "Notch filter applied! Periodic noise removed.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Notch filter failed: {str(e)}")

    def _apply_notch_to_channel(self, channel):
        """Apply notch filter to a single channel"""
        # FFT
        ft = np.fft.fft2(channel)
        ft_shift = np.fft.fftshift(ft)
        magnitude = np.abs(ft_shift)
        
        # Auto-detect notch points
        notch_pairs = self.detect_notch_points(magnitude, threshold_percentile=99.9, min_distance=10)
        
        if len(notch_pairs) == 0:
            messagebox.showinfo("Info", "No significant periodic noise detected in this image.")
            return channel
        
        # Calculate anti-notch pairs (symmetric points)
        center_u, center_v = channel.shape[0]//2, channel.shape[1]//2
        anti_notch_pairs = []
        for notch in notch_pairs:
            anti_notch_u = center_u - (notch[0] - center_u)
            anti_notch_v = center_v - (notch[1] - center_v)
            anti_notch_pairs.append((anti_notch_u, anti_notch_v))
        
        # Generate mask
        D0k = 5  # Notch radius
        n = 2    # Filter order
        mask = self.gen_mask(channel.shape, notch_pairs, anti_notch_pairs, D0k, n)
        
        # Apply filter
        filtered_ft = ft_shift * mask
        filtered_ft_shift = np.fft.ifftshift(filtered_ft)
        filtered_channel = np.fft.ifft2(filtered_ft_shift)
        filtered_channel = np.real(filtered_channel)
        filtered_channel = cv2.normalize(filtered_channel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return filtered_channel
    
    def show_hint(self):
        if self.game_started and self.current_level in self.level_clues:
            hint = self.level_clues[self.current_level]["hint"]
            messagebox.showinfo("Hint", hint)
    
    def submit_answer(self):
        if not self.game_started:
            messagebox.showwarning("Warning", "Please start the game first!")
            return
            
        answer=simpledialog.askstring("Submit Answer", 
                                       f"Level {self.current_level}: {self.level_clues[self.current_level]['question']}")
        
        if answer is None: 
            return
            
        correct_answer=self.level_clues[self.current_level]["answer"].lower()
        user_answer=answer.lower().strip()
        
        import re
        user_answer=re.sub(r'[^\w\s]', '', user_answer)
        user_answer=re.sub(r'\s+', ' ', user_answer)
        
        if correct_answer in user_answer or user_answer in correct_answer:
            points = 100
            self.score += points
            self.score_label.config(text=str(self.score))
            
            messagebox.showinfo("Correct!", 
                              f"Excellent detective work! You found: '{answer}'\n+{points} points!\n\nMoving to next level...")
        else:
            messagebox.showinfo("Incorrect Answer", 
                              f"That's not correct. The answer was: '{self.level_clues[self.current_level]['answer']}'\n\nNo points awarded, but moving to next level...")
        
        if self.current_level<self.max_levels:
            self.current_level+= 1
            self.level_label.config(text=str(self.current_level))
            self.load_level()
        else:
            # Game completed after level 3
            max_score = self.max_levels * 100
            
            messagebox.showinfo("Game Complete!", 
                              f"Congratulations Detective!\n\nFinal Score: {self.score} / {max_score}\n\nClick 'Start Game' to play again!")
            self.end_game()

if __name__ == "__main__":
    root = tk.Tk()
    game = ClueHuntingGame(root)
    root.mainloop()