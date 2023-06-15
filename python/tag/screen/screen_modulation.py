import tkinter as tk

color = "white"
freq = 10 # Hz
switching_time = int(1 / freq / 2 * 1000)

def switch_color():
    current_color = window.cget("bg")
    new_color = "black" if current_color == color else color
    window.configure(bg=new_color)
    window.after(switching_time, switch_color)

window = tk.Tk()
window.geometry("200x200")
window.configure(bg=color)

switch_color()

window.mainloop()
