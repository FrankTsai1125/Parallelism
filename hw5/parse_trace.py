import json
import csv
import sys
import os

def parse_trace(input_file, output_csv):
    try:
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found")
            return

        kernel_stats = {}
        
        # Determine file type by extension
        is_json = input_file.endswith('.json')
        
        if is_json:
            with open(input_file, 'r') as f:
                data = json.load(f)
                events = data if isinstance(data, list) else data.get('traceEvents', [])
                for event in events:
                    if event.get('ph') == 'X':
                        name = event.get('name', 'Unknown')
                        if 'Kernel' in name or 'kernel' in name or 'Memcpy' in name:
                            dur = event.get('dur', 0)
                            # Convert us to ns if needed, but usually JSON is in us
                            # Let's standardize on ns for output if possible, or just keep source unit
                            if name not in kernel_stats:
                                kernel_stats[name] = {'count': 0, 'total_time': 0.0}
                            kernel_stats[name]['count'] += 1
                            kernel_stats[name]['total_time'] += dur
        else:
            # Assume CSV from rocprofv2 --kernel-trace
            # Typical Headers: "Index","KernelName","SharedMemBytes","GridWorkgroupSizeX"... "StartTimestamp","EndTimestamp","DurationNs"
            # We need to be flexible with headers
            with open(input_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try to find the Name column
                    name = row.get('KernelName') or row.get('Name') or row.get('Kernel_Name')
                    if not name: continue
                    
                    # Try to find Duration column
                    dur_str = row.get('DurationNs') or row.get('Duration') or row.get('Duration(ns)')
                    if not dur_str: continue
                    
                    try:
                        dur = float(dur_str)
                    except ValueError:
                        continue
                        
                    if name not in kernel_stats:
                        kernel_stats[name] = {'count': 0, 'total_time': 0.0}
                    kernel_stats[name]['count'] += 1
                    kernel_stats[name]['total_time'] += dur

        # Calculate stats
        total_execution_time = sum(s['total_time'] for s in kernel_stats.values())
        
        stats_list = []
        for name, stats in kernel_stats.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            percent = (stats['total_time'] / total_execution_time * 100) if total_execution_time > 0 else 0
            stats_list.append({
                'Name': name,
                'Count': stats['count'],
                'TotalTime': stats['total_time'], 
                'AvgTime': avg_time,
                'Percentage': percent
            })

        # Sort by Total Time descending
        stats_list.sort(key=lambda x: x['TotalTime'], reverse=True)

        # Write to CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Name', 'Count', 'TotalTime', 'AvgTime', 'Percentage'])
            writer.writeheader()
            writer.writerows(stats_list)
            
        print(f"Successfully wrote stats to {output_csv}")

    except Exception as e:
        print(f"Error parsing trace: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 parse_trace.py <input_file> <output_csv>")
        sys.exit(1)
    
    parse_trace(sys.argv[1], sys.argv[2])
