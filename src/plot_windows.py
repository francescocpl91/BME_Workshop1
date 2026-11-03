"""Desktop display of final-analysis PNGs, separate from headless chart generation."""
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading


def _gallery(folder):
    """Use only PNGs listed by the published report, never neighboring run folders."""
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest,dict) or not manifest.get("complete") or not isinstance(manifest.get("files"),dict):
        raise ValueError("Incomplete report")
    return sorted(name for name in manifest["files"]
                  if Path(name).name == name and name.endswith(".png") and (folder/name).is_file())


def launch(folder, families):
    """One window per requested chart family; every window can browse the full gallery."""
    if not families:
        return False
    files = _gallery(folder)
    initial = []
    for family in families:
        candidates = [name for name in files if name[:-4].rsplit("_",1)[0] == family]
        if candidates:
            initial.append(candidates[0])
    if not initial:
        return False
    child = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), str(folder), *initial],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                             creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    # The viewer controls its own GUI event loop. Wait only for startup so a
    # missing Tk installation or inaccessible display gives run.py a useful fallback.
    ready = queue.Queue()
    def read_ready():
        with child.stdout as stream:
            ready.put(stream.readline().strip())
    threading.Thread(target=read_ready,daemon=True).start()
    try:
        opened = ready.get(timeout=12) == "ready"
    except queue.Empty:
        opened = False
    if not opened and child.poll() is None:
        child.terminate()
        child.wait()
    return opened


def _window(root, folder, names, selected):
    """Display the existing image at native resolution with scrollbars for long path panels."""
    import tkinter as tk
    from tkinter import ttk
    win = tk.Toplevel(root)
    win.title("Portfolio analysis | "+selected)
    choice = ttk.Combobox(win,values=names,state="readonly",width=55)
    choice.set(selected)
    choice.grid(row=0,column=0,sticky="ew")
    canvas = tk.Canvas(win,highlightthickness=0,background="white")
    canvas.grid(row=1,column=0,sticky="nsew")
    for orient,row,col,sticky,command in (("vertical",1,1,"ns",canvas.yview),
                                         ("horizontal",2,0,"ew",canvas.xview)):
        bar=ttk.Scrollbar(win,orient=orient,command=command)
        bar.grid(row=row,column=col,sticky=sticky)
        canvas.configure(**{("y" if orient=="vertical" else "x")+"scrollcommand":bar.set})
    win.rowconfigure(1,weight=1)
    win.columnconfigure(0,weight=1)
    def display(event=None):
        filename=choice.get()
        im=tk.PhotoImage(master=win,file=str(folder/filename))
        canvas.image=im  # Tk holds a reference by name only.
        canvas.delete("all")
        canvas.create_image(0,0,image=im,anchor="nw")
        canvas.configure(width=min(im.width(),int(root.winfo_screenwidth()*.8)),
                         height=min(im.height(),int(root.winfo_screenheight()*.72)),
                         scrollregion=(0,0,im.width(),im.height()))
        canvas.yview_moveto(0); canvas.xview_moveto(0)
        win.title("Portfolio analysis | "+filename)
    choice.bind("<<ComboboxSelected>>",display)
    display()
    def close():
        win.destroy()
        if not root.winfo_children():
            root.destroy()
    win.protocol("WM_DELETE_WINDOW",close)
    return win


if __name__ == "__main__":
    import tkinter as tk
    folder=Path(sys.argv[1])
    names=_gallery(folder)
    root=tk.Tk()
    root.withdraw()
    for selected in sys.argv[2:]:
        if selected in names:
            _window(root,folder,names,selected)
    if not root.winfo_children():
        root.destroy()
        sys.exit(1)
    root.update()
    print("ready",flush=True)
    root.mainloop()
