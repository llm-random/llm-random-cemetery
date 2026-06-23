import json

file_path = "/Users/boro/code/llm-random/nano/flops_comparison.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

definition_updated = False
calling_updated = False

for cell in data.get("cells", []):
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        source_text = "".join(source_lines)
        
        # Check if this is the cell containing the definition
        if "def plot_flops_comparison(" in source_text:
            print("Found definition cell!")
            
            # 1. Remove the old legend call on ax1
            old_legend_call = '    ax1.legend(loc="upper left")\n'
            # 2. Add the combined legend after ax2.set_ylim
            old_ylim_call = '    ax2.set_ylim(0, 1.0)\n'
            new_ylim_call = (
                '    ax2.set_ylim(0, 1.0)\n'
                '\n'
                '    # Combine legends from ax1 and ax2\n'
                '    lines1, labels1 = ax1.get_legend_handles_labels()\n'
                '    lines2, labels2 = ax2.get_legend_handles_labels()\n'
                '    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")\n'
            )
            
            if old_legend_call in source_text and old_ylim_call in source_text:
                new_text = source_text.replace(old_legend_call, "")
                new_text = new_text.replace(old_ylim_call, new_ylim_call)
                
                # Check for TODO in the same cell
                if "# todo add red line to the legend" in new_text:
                    print("Found calling/TODO section in the same cell!")
                    new_text = new_text.replace("\n\n# todo add red line to the legend", "")
                    new_text = new_text.replace("\n# todo add red line to the legend", "")
                    new_text = new_text.replace("# todo add red line to the legend", "")
                    calling_updated = True
                
                new_lines = new_text.splitlines(True)
                cell["source"] = new_lines
                definition_updated = True
                print("Definition cell successfully modified!")
            else:
                print("Could not find legend call or ylim call in definition cell!")

if definition_updated and calling_updated:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print("Notebook successfully updated and saved!")
else:
    print(f"Error: definition_updated={definition_updated}, calling_updated={calling_updated}")
