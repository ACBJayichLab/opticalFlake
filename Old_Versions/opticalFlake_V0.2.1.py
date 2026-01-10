
#pip install pillow numpy matplotlib

import tkinter as tk
#Buttons don't always register fast clicks, slower clicks seem to work better
from PIL import Image, ImageGrab, ImageTk, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import platform
import sys

# USAGE INSTRUCTIONS
# Position window over relevant image
# Click "Capture Screenshot"
# Set up background polygon: "Start BG Polygon", Click at least 3 points, "Finish BG Polygon"
# Enter averaging width (can be done after "start sample linecut but averaging lines won't be displayed")
# Click "Start Sample Linecut"
# Click two points
# Click Averaging Width

#Known/Suspected bugs
# If the background 


def create_transparent_window():
    # Create the main window
    root = tk.Tk()
    root.title("Optical Flake Thickness Characterizer")

    # Configure the window size
    window_width = 1200
    window_height = 1000
    root.geometry(f"{window_width}x{window_height}")

    # Make the window background transparent (platform-specific)
    transparent_color = "white"  # Use a specific color for Windows
    root.configure(bg=transparent_color)
    
    # Handle platform-specific transparency
    system = platform.system()
    if system == "Windows":
        root.attributes("-transparentcolor", transparent_color)
    # On macOS, tkinter doesn't support color-based transparency like Windows
    # The window will show with a white background, but you can still position it over your image
    # On Linux, transparency handling varies by window manager

    # Create a canvas to draw on
    canvas = tk.Canvas(root, width=window_width, height=window_height, bg=transparent_color, highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)


    global points
    points = []  # Store points clicked by the user
    global average_background
    average_background = (255,255,255)
    global lineStart
    lineStart=(0,0,0)
    global lineEnd
    lineEnd=(1,1,1)
    global lineCut_click_num
    lineCut_click_num=0
    global lineCut_id, lineCut_upper_id, lineCut_lower_id
    global screenshot

    root.update()
    def start_linecut():
        root.update_idletasks()
        global lineStart, lineEnd, lineStart_id, lineEnd_id, lineCut_id, lineCut_lower_id, lineCut_upper_id
        if 'lineStart_id' in globals() and lineStart_id:  # Check if there's an existing polygon
            canvas.delete(lineStart_id)  # Remove the old polygon from the canvas
            lineStart_id = None  # Reset the polygon_id
        if 'lineEnd_id' in globals() and lineEnd_id:  # Check if there's an existing polygon
            canvas.delete(lineEnd_id)  # Remove the old polygon from the canvas
            lineEnd_id = None  # Reset the polygon_id
        if 'lineCut_id' in globals() and lineCut_id:  # Check if there's an existing polygon
            canvas.delete(lineCut_id)  # Remove the old polygon from the canvas
            lineCut_id = None  # Reset the polygon_id
        if 'lineCut_lower_id' in globals() and lineCut_lower_id:  # Check if there's an existing polygon
            canvas.delete(lineCut_lower_id)  # Remove the old polygon from the canvas
            lineCut_lower_id = None  # Reset the polygon_id
        if 'lineCut_upper_id' in globals() and lineCut_upper_id:  # Check if there's an existing polygon
            canvas.delete(lineCut_upper_id)  # Remove the old polygon from the canvas
            lineCut_upper_id = None  # Reset the polygon_id
        global lineCut_click_num
        lineCut_click_num=0
        print("Looking for clicks")
        """Enable click capturing."""
        canvas.bind("<Button-1>", on_linecut_click)
        start_linecut_button.config(state=tk.DISABLED)  # Disable the start button to avoid multiple activations
        root.focus_force()

    def on_linecut_click(event):
        root.update_idletasks()
        print("Click!")
        """Handle click events and record points."""
        global lineStart, lineStart_id, lineEnd, lineEnd_id, lineCut_id, lineCut_click_num, lineCut_lower_id, lineCut_upper_id
        if(lineCut_click_num==0):
            lineStart=(event.x, event.y)
        # Draw a small dot to mark the click
            lineStart_id=canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill="black", outline="black")
            lineCut_click_num=1
        elif(lineCut_click_num==1):
            lineEnd=(event.x, event.y)
        # Draw a small dot to mark the click
            lineEnd_id=canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill="black", outline="black")
            lineCut_click_num=2
            canvas.unbind("<Button-1>")
            lineCut_id=canvas.create_line(lineStart[0],lineStart[1],lineEnd[0],lineEnd[1], fill="white",width='1')
            avg_width = int(averaging_entry.get())
            if avg_width>1:
                newx1,newy1,nexx2,newy2=offset_parallel_line(lineStart[0],lineStart[1],lineEnd[0],lineEnd[1],math.ceil(avg_width/2))
                lineCut_lower_id=canvas.create_line(newx1,newy1,nexx2,newy2, fill="yellow",width='1')
                newx1,newy1,nexx2,newy2=offset_parallel_line(lineStart[0],lineStart[1],lineEnd[0],lineEnd[1],-1*math.ceil(avg_width/2))
                lineCut_upper_id=canvas.create_line(newx1,newy1,nexx2,newy2, fill="yellow",width='1')
            start_linecut_button.config(state=tk.NORMAL)
            calculate_contrast()
        root.update()
        root.focus_force()


    def start_clicks():
        root.update_idletasks()
        global points, polygon_id
        points = []
        if 'polygon_id' in globals() and polygon_id:  # Check if there's an existing polygon
            canvas.delete(polygon_id)  # Remove the old polygon from the canvas
            polygon_id = None  # Reset the polygon_id
        print("Looking for clicks")
        """Enable click capturing."""
        canvas.bind("<Button-1>", on_click)
        start_button.config(state=tk.DISABLED)  # Disable the start button to avoid multiple activations
        root.focus_force()



    def on_click(event):
        root.update_idletasks()
        print("Click!")
        """Handle click events and record points."""
        points.append((event.x, event.y))
        # Draw a small dot to mark the click
        #canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill="black", outline="black")
        root.update()
        root.focus_force()

    def finish_polygon():
        root.update_idletasks()
        global polygon_id 
        """Stop click capturing and draw the polygon."""
        canvas.unbind("<Button-1>")  # Stop capturing clicks
        start_button.config(state=tk.NORMAL)  # Re-enable the start button
        if len(points) > 2:  # Draw a polygon if there are at least 3 points
            polygon_id = canvas.create_polygon(points, outline="green", fill="", width=2)
            print("Setting Background")
        root.update()
        root.focus_force()

    def set_background_to_screenshot():
        """Replace the canvas background with a screenshot of the area behind the window."""
        # Get the window's position
        root.update_idletasks()
        x = root.winfo_rootx()
        y = root.winfo_rooty()
        width = x + root.winfo_width()
        height = y + root.winfo_height()

        # Capture the area behind the window
        global screenshot
        screenshot = ImageGrab.grab(bbox=(x, y, width, height))
        screenshot_tk = ImageTk.PhotoImage(screenshot)

        # Display the screenshot as the background
        canvas.create_image(0, 0, anchor="nw", image=screenshot_tk)
        canvas.image = screenshot_tk  # Keep a reference to avoid garbage collection
        
        root.resizable(False, False)
        print("Setting Background")
        root.update()
        root.focus_force()

    def clear_background():
        """Remove the screenshot background and reset the canvas to transparent."""
        print("Clearing Background")
        canvas.delete("all")  # Clear all items from the canvas
        root.resizable(True, True)
        root.update()
        root.focus_force()
    def capture_canvas_as_image(canvas):
        """Capture the canvas content as an image."""
        x = root.winfo_rootx() + canvas.winfo_x()
        y = root.winfo_rooty() + canvas.winfo_y()
        width = x + canvas.winfo_width()
        height = y + canvas.winfo_height()

        return ImageGrab.grab(bbox=(x, y, width, height))
    def create_polygon_mask(image_size, points):
        """Create a mask image where the polygon is white and the rest is black."""
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, fill=255)
        return mask

    def calculate_average_color(image, mask):
        """Calculate the average color within the masked region of an image."""
        masked_image = Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)
        pixels = list(masked_image.getdata())
        valid_pixels = [pixel for pixel in pixels if pixel != (0, 0, 0)]
        if valid_pixels:
            r = sum(p[0] for p in valid_pixels) // len(valid_pixels)
            g = sum(p[1] for p in valid_pixels) // len(valid_pixels)
            b = sum(p[2] for p in valid_pixels) // len(valid_pixels)
            return (r, g, b)
        else:
            return (0, 0, 0)

    def get_average_color():
        """Calculate and print the average color of the polygon area."""
        global polygon_id, lineCut_id, lineCut_upper_id, lineCut_lower_id, lineStart_id,lineEnd_id
        canvas.itemconfig(polygon_id, outline=transparent_color)
        canvas.itemconfig(lineCut_id, fill=transparent_color)
        if 'lineCut_upper_id' in globals() and lineCut_upper_id:
            canvas.itemconfig(lineCut_upper_id, fill=transparent_color)
        if 'lineCut_lower_id' in globals() and lineCut_lower_id:
            canvas.itemconfig(lineCut_lower_id, fill=transparent_color)
        canvas.itemconfig(lineStart_id, fill=transparent_color)
        canvas.itemconfig(lineEnd_id, fill=transparent_color)
        root.update()
        root.update_idletasks()
        image = capture_canvas_as_image(canvas)
        mask = create_polygon_mask(image.size, points)
        avg_color = calculate_average_color(image, mask)
        global average_background
        average_background=avg_color
        print(f"Average color: {avg_color}")
        canvas.itemconfig(polygon_id, outline="red")
        canvas.itemconfig(lineCut_id, fill="white")
        if 'lineCut_upper_id' in globals() and lineCut_upper_id:
            canvas.itemconfig(lineCut_upper_id, fill="yellow")
        if 'lineCut_lower_id' in globals() and lineCut_lower_id:
            canvas.itemconfig(lineCut_lower_id, fill="yellow")
        canvas.itemconfig(lineStart_id, fill="black")
        canvas.itemconfig(lineEnd_id, fill="black")
        root.update()
        return average_background
    
    def get_line_coordinates(x1, y1, x2, y2):
        """
        Bresenham's Line Algorithm to generate points between two points (x1, y1) and (x2, y2).
        This version checks if the line has a steep slope (dy > dx) and swaps x and y if necessary.
        
        Parameters:
            x1, y1: Coordinates of the first point.
            x2, y2: Coordinates of the second point.
            
        Returns:
            A list of coordinates of the points on the line.
        """
        points = []
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        steep = dy > dx  # Check if the slope is steep

        if steep:
            # Swap x and y if the slope is steep
            x1, y1 = y1, x1
            x2, y2 = y2, x2
            dx, dy = dy, dx
        
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            if steep:
                points.append((y1, x1))  # Swap x and y back for steep lines
            else:
                points.append((x1, y1))
            
            if x1 == x2 and y1 == y2:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        
        return points

    def get_line_rgb_values(image, x1, y1, x2, y2):
        """Get RGB values of pixels along a line between two coordinates."""
        # Get line coordinates
        line_coordinates = get_line_coordinates(x1, y1, x2, y2)
    
       # Extract RGB values for each coordinate
        red_values = []
        green_values = []
        blue_values = []

        for x, y in line_coordinates:
            r, g, b = image.getpixel((x, y))
            red_values.append(r)
            green_values.append(g)
            blue_values.append(b)

        np_red = np.array(red_values)
        np_green = np.array(green_values)
        np_blue = np.array(blue_values)
        return np_red, np_green, np_blue

    def calculate_contrast():
        global screenshot, lineStart, lineEnd
        avg_width = int(averaging_entry.get())
        avg_bg=get_average_color()
        redLine, greenLine, blueLine = get_line_rgb_values(screenshot,lineStart[0],lineStart[1],lineEnd[0],lineEnd[1])
        redLineNorm=(redLine-avg_bg[0])/avg_bg[0]
        greenLineNorm=(greenLine-avg_bg[1])/avg_bg[1]
        blueLineNorm=(blueLine-avg_bg[2])/avg_bg[2]
        running_sum_red = np.zeros_like(redLineNorm)
        running_sum_green = np.zeros_like(greenLineNorm)
        running_sum_blue = np.zeros_like(blueLineNorm)
        redLineNorm_avg = np.zeros_like(redLineNorm)
        greenLineNorm_avg = np.zeros_like(greenLineNorm)
        blueLineNorm_avg = np.zeros_like(blueLineNorm)
        running_sum_red += redLineNorm
        running_sum_green += greenLineNorm
        running_sum_blue += blueLineNorm
        numRuns=1
        for x in range(1,math.ceil(avg_width/2)+1):
            #print(math.ceil(avg_width/2))
            print(x)
            if x==1:
                redLineNorm_avg=redLineNorm
                greenLineNorm_avg=greenLineNorm
                blueLineNorm_avg=blueLineNorm
                pass
            else:
                newx1,newy1,nexx2,newy2=offset_parallel_line(lineStart[0],lineStart[1],lineEnd[0],lineEnd[1],x)
                redLine, greenLine, blueLine = get_line_rgb_values(screenshot,newx1,newy1,nexx2,newy2)
                redLineNorm=(redLine-avg_bg[0])/avg_bg[0]
                greenLineNorm=(greenLine-avg_bg[1])/avg_bg[1]
                blueLineNorm=(blueLine-avg_bg[2])/avg_bg[2]
                running_sum_red += redLineNorm
                running_sum_green += greenLineNorm
                running_sum_blue += blueLineNorm
                numRuns +=1

                newx1,newy1,nexx2,newy2=offset_parallel_line(lineStart[0],lineStart[1],lineEnd[0],lineEnd[1],-1*x)
                redLine, greenLine, blueLine = get_line_rgb_values(screenshot,newx1,newy1,nexx2,newy2)
                redLineNorm=(redLine-avg_bg[0])/avg_bg[0]
                greenLineNorm=(greenLine-avg_bg[1])/avg_bg[1]
                blueLineNorm=(blueLine-avg_bg[2])/avg_bg[2]
                running_sum_red += redLineNorm
                running_sum_green += greenLineNorm
                running_sum_blue += blueLineNorm
                numRuns +=1

                redLineNorm_avg=running_sum_red/numRuns
                greenLineNorm_avg=running_sum_green/numRuns
                blueLineNorm_avg=running_sum_blue/numRuns
        plot_arrays_in_tkinter(redLineNorm_avg,greenLineNorm_avg,blueLineNorm_avg)


    def offset_parallel_line(x1, y1, x2, y2, L):
        """
        Given a line formed by (x1, y1) and (x2, y2), create a new parallel line offset by L.

        Parameters:
            x1, y1: Coordinates of the first point of the original line.
            x2, y2: Coordinates of the second point of the original line.
            L: Offset distance for the parallel line.

        Returns:
            (new_x1, new_y1, new_x2, new_y2): Coordinates of the new parallel line.
        """
        # Calculate the direction vector of the line
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate the length of the original line segment
        length = math.sqrt(dx**2 + dy**2)
        
        # Normalize the direction vector to get the unit perpendicular vector
        unit_perpendicular_x = -dy / length
        unit_perpendicular_y = dx / length

        # Calculate the offset for the parallel line
        offset_x = L * unit_perpendicular_x
        offset_y = L * unit_perpendicular_y

        # Compute the new parallel line's endpoints
        new_x1 = x1 + offset_x
        new_y1 = y1 + offset_y
        new_x2 = x2 + offset_x
        new_y2 = y2 + offset_y
        
        new_x1 = np.round(new_x1)
        new_y1 = np.round(new_y1)
        new_x2 = np.round(new_x2)
        new_y2 = np.round(new_y2)


        return new_x1, new_y1, new_x2, new_y2

    def plot_arrays_in_tkinter(red_array, green_array, blue_array):
        """
        Create a new Tkinter window and display three plots for the input arrays.

        Parameters:
        - red_array: NumPy array for the red channel.
        - green_array: NumPy array for the green channel.
        - blue_array: NumPy array for the blue channel.
        """
        if not (len(red_array) == len(green_array) == len(blue_array)):
            raise ValueError("All arrays must have the same length.")
    
        # Create a new Tkinter window
        global window_id
        if 'window_id' in globals() and window_id:
            window_id.destroy()
            print("Trying to close")
        window = tk.Tk()
        window_id=window
        window.title("RGB Channel Plots")

        # Create a Matplotlib figure
        #fig, axes = plt.subplots(3, 1, figsize=(8, 6))
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))

        # Plot the Red array
        axes[0].plot(red_array, color='red')
        axes[0].set_title("Red Channel")
        axes[0].set_xlabel("Pixels")
        axes[0].set_ylabel("Contrast (%)")
        axes[0].grid()

        # Plot the Green array
        axes[1].plot(green_array, color='green')
        axes[1].set_title("Green Channel")
        axes[1].set_xlabel("Pixels")
        axes[1].set_ylabel("Contrast (%)")
        axes[1].grid()
        # Plot the Blue array
        #axes[2].plot(blue_array, color='blue')
        #axes[2].set_title("Blue Channel")
        #axes[2].set_xlabel("Index")
        #axes[2].set_ylabel("Value")

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 1])

        # Embed the Matplotlib figure in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Run the Tkinter main loop
        window.mainloop()


    button_frame = tk.Frame(root, bg="white")
    button_frame.place(relx=1.0, rely=0.0, anchor="ne", width=160, height=360)
    # Create buttons
    start_button = tk.Button(button_frame, text="Start BG Polygon", command=start_clicks, bg="white")
    finish_button = tk.Button(button_frame, text="Finish BG Polygon", command=finish_polygon, bg="white")
    screenshot_button = tk.Button(button_frame, text="Capture Screenshot", command=set_background_to_screenshot, bg="white")
    clear_button = tk.Button(button_frame, text="Clear Screenshot", command=clear_background, bg="white")
    background_button=tk.Button(button_frame, text="Get Average BG Color", command=get_average_color, bg="white")
    start_linecut_button = tk.Button(button_frame, text="Start Sample Linecut", command=start_linecut, bg="white")
    calculate_button = tk.Button(button_frame, text="Calculate Contrast", command=calculate_contrast, bg="white")
    averaging_label=tk.Label(button_frame, text="Averaging Width", bg="white")
    averaging_width=tk.StringVar(value="10")
    averaging_entry=tk.Entry(button_frame,textvariable=averaging_width, bg="white")

    # Place buttons at the top-right corner
    screenshot_button.place(relx=1.0, y=10, anchor="ne", x=-10)
    clear_button.place(relx=1.0, y=50, anchor="ne", x=-10)
    start_button.place(relx=1.0, y=90, anchor="ne", x=-10)
    finish_button.place(relx=1.0, y=130, anchor="ne", x=-10)
    background_button.place(relx=1.0, y=170, anchor="ne", x=-10)
    start_linecut_button.place(relx=1.0, y=210, anchor="ne", x=-10)
    calculate_button.place(relx=1.0, y=250, anchor="ne", x=-10)
    averaging_label.place(relx=1.0, y=290, anchor="ne", x=-10)
    averaging_entry.place(relx=1.0, y=330, anchor="ne", x=-10)

    # Position the window at the center of the screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    root.mainloop()

create_transparent_window()
