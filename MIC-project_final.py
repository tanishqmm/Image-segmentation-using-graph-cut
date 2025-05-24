import numpy as np
import cv2
import maxflow

class InteractiveGraphCutSegmenter:
    def __init__(self, image_path):
        """Initialize the interactive graph cut segmenter."""
        # Load image
        self.image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        
        if self.image is None:
            raise ValueError("Could not load image from path")
        self.image = cv2.resize(self.image, (512, 512), interpolation=cv2.INTER_AREA)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        
        # Initialize seeds and segmentation
        self.object_seeds = set()
        self.background_seeds = set()
        self.segmentation = None
        self.display_mode = 1  # 1=overlay, 2=white_bg, 3=black_bg, 4=blur_bg
        
        # Drawing state
        self.drawing = False
        self.current_seed_type = None  # 'object' or 'background'
        
        # Colors
        self.OBJECT_COLOR = (255, 0, 0)    # Red
        self.BACKGROUND_COLOR = (0, 0, 255) # Blue
        self.SEED_RADIUS = 3
        
        # Parameters
        self.lambda_ = 0.001
        self.sigma = 10
        
        # Create window and set mouse callback
        cv2.namedWindow("Interactive Graph Cut")
        cv2.setMouseCallback("Interactive Graph Cut", self._mouse_callback)
        
        # Display instructions
        print("Interactive Graph Cut Segmentation")
        print("Press 'o' then left-click+drag: Add object seeds")
        print("Press 'b' then left-click+drag: Add background seeds")
        print("Press 's' to perform segmentation")
        print("Press 'c' to clear seeds")
        print("Press '1-4' to change display mode")
        print("Press 'q' to quit")
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for seed placement with dragging."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self._add_seed(x, y, self.current_seed_type)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._add_seed(x, y, self.current_seed_type)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            
        self._update_display()
    
    def _add_seed(self, x, y, seed_type):
        """Add a seed point at (x,y) of the specified type."""
        if 0 <= x < self.width and 0 <= y < self.height:
            if seed_type == 'object':
                self.object_seeds.add((x, y))
            elif seed_type == 'background':
                self.background_seeds.add((x, y))
    
    def clear_seeds(self):
        """Clear all seed points."""
        self.object_seeds = set()
        self.background_seeds = set()
        self.segmentation = None
        self.current_seed_type = None
        self.drawing = False

    def _create_display(self):
        """Create the display image based on current mode."""
        display = self.image.copy()
        
        # Draw seeds
        for x, y in self.object_seeds:
            cv2.circle(display, (x, y), self.SEED_RADIUS, self.OBJECT_COLOR, -1)
        for x, y in self.background_seeds:
            cv2.circle(display, (x, y), self.SEED_RADIUS, self.BACKGROUND_COLOR, -1)
        
        if self.segmentation is not None:
            mask = self.segmentation.astype(bool)
            
            if self.display_mode == 1:  # Uniform green overlay
                overlay = display.copy()
                overlay[mask] = [0, 255, 0]  # Green
                display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
                
            elif self.display_mode == 2:  # White background
                result = display.copy()
                result[mask] = [255, 255, 255]
                display = result
                
            elif self.display_mode == 3:  # Black background
                result = display.copy()
                result[mask] = [0, 0, 0]
                display = result
                
            elif self.display_mode == 4:  # Blurred background
                blurred = cv2.GaussianBlur(display, (51, 51), 0)
                display[~mask] = self.image[~mask]
                display[mask] = blurred[mask]
        
        return cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    
    def _update_display(self):
        """Update the display window."""
        cv2.imshow("Interactive Graph Cut", self._create_display())
    
    def compute_histograms(self):
        """Compute histograms for object and background seeds."""
        if len(self.object_seeds) == 0 or len(self.background_seeds) == 0:
            return
            
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        object_pixels = [gray_img[y, x] for x, y in self.object_seeds]
        background_pixels = [gray_img[y, x] for x, y in self.background_seeds]
        
        self.object_hist = np.histogram(object_pixels, bins=64, range=(0, 256))[0]
        self.background_hist = np.histogram(background_pixels, bins=64, range=(0, 256))[0]
        
        self.object_hist = self.object_hist / (np.sum(self.object_hist) + 1e-10)
        self.background_hist = self.background_hist / (np.sum(self.background_hist) + 1e-10)
    
    def regional_term(self, intensity, label):
        """Compute regional term (negative log likelihood) for a pixel."""
        if self.object_hist is None or self.background_hist is None:
            return 0
            
        bin_idx = min(int(intensity * 64 / 256), 63)
        prob = self.object_hist[bin_idx] if label == "obj" else self.background_hist[bin_idx]
        return -np.log(prob + 1e-10)
    
    def boundary_term(self, p, q):
        """Compute boundary term between two neighboring pixels."""
        diff = np.linalg.norm(self.image[p[1], p[0]] - self.image[q[1], q[0]])
        dx, dy = p[0] - q[0], p[1] - q[1]
        dist = np.sqrt(dx*dx + dy*dy)
        return np.exp(-(diff**2) / (2 * self.sigma**2)) / dist
    
    def build_graph(self):
        """Build the graph for max-flow/min-cut computation."""
        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes((self.height, self.width))
        
        # K = 1e10 + 8 * np.exp(-0 / (2 * self.sigma**2))
        K = np.inf
        for y in range(self.height):
            for x in range(self.width):
                p = (x, y)
                
                if p in self.object_seeds:
                    g.add_tedge(nodeids[y, x], K, 0)
                elif p in self.background_seeds:
                    g.add_tedge(nodeids[y, x], 0, K)
                else:
                    intensity = np.mean(self.image[y, x])
                    r_obj = self.regional_term(intensity, "obj")
                    r_bkg = self.regional_term(intensity, "bkg")
                    g.add_tedge(nodeids[y, x], self.lambda_ * r_bkg, self.lambda_ * r_obj)
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            weight = self.boundary_term(p, (nx, ny))
                            g.add_edge(nodeids[y, x], nodeids[ny, nx], weight, weight)
        
        return g, nodeids
    
    def segment(self):
        """Perform the segmentation using graph cuts."""
        if len(self.object_seeds) == 0 or len(self.background_seeds) == 0:
            print("Error: Need both object and background seeds")
            return False
            
        self.compute_histograms()
        g, nodeids = self.build_graph()
        flow = g.maxflow()
        print(f"Max flow: {flow}")
        
        self.segmentation = np.zeros((self.height, self.width), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                if g.get_segment(nodeids[y, x]):
                    self.segmentation[y, x] = 1
        
        return True
    
    def run(self):
        """Run the interactive segmentation application."""
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('o'):  # Start marking object seeds
                self.current_seed_type = 'object'
                print("Marking object seeds - click and drag")
            
            elif key == ord('b'):  # Start marking background seeds
                self.current_seed_type = 'background'
                print("Marking background seeds - click and drag")
            
            elif key == ord('s'):  # Perform segmentation
                if self.segment():
                    self._update_display()
                else:
                    print("Please add both object and background seeds first")
            
            elif key == ord('c'):  # Clear seeds
                self.clear_seeds()
                self._update_display()
            
            elif key == ord('1'):  # Overlay mode
                self.display_mode = 1
                self._update_display()
            
            elif key == ord('2'):  # White background
                self.display_mode = 2
                self._update_display()
            
            elif key == ord('3'):  # Black background
                self.display_mode = 3
                self._update_display()
            
            elif key == ord('4'):  # Blurred background
                self.display_mode = 4
                self._update_display()
            
            elif key == ord('q'):  # Quit
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    segmenter = InteractiveGraphCutSegmenter("crowd.png")
    segmenter.run()