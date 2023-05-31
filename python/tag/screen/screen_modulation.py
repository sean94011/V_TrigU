import tkinter as tk

def switch_color():
    current_color = window.cget("bg")
    new_color = "black" if current_color == "red" else "red"
    window.configure(bg=new_color)
    window.after(1000, switch_color)

window = tk.Tk()
window.geometry("200x200")
window.configure(bg="red")

switch_color()

window.mainloop()
