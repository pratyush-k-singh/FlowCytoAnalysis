import subprocess

def run_script(script_path):
    """Run a script and wait for it to finish."""
    process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode != 0:
        print(f"Error running {script_path}: {err.decode('utf-8')}")
    else:
        print(f"Output of {script_path}: {out.decode('utf-8')}")

if __name__ == "__main__":
    # Full paths to the scripts to run
    scripts = [
        r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\GRU\GRU_Model.py',
        r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\TCN\TCN_Model.py',
        r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\TFT\TFT_Model.py',
        r'C:\Users\praty\cytoflow\Codeset\Model_Testing\Models\Totem\Totem_Model.py'
    ]

    # Run scripts in parallel
    processes = []
    for script in scripts:
        print(f"Starting {script}...")
        p = subprocess.Popen(['python', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(p)

    # Wait for all scripts to finish
    for p in processes:
        p.wait()
        out, err = p.communicate()
        if p.returncode != 0:
            print(f"Error: {err.decode('utf-8')}")
        else:
            print(f"Output: {out.decode('utf-8')}")

    print("All scripts finished.")
