import os
import re
import pandas as pd
import glob
import pickle

def parse_iv_file(filepath):
    dataset = []
    current_record = {}
    in_batch_params = False
    in_data_table = False
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            
            # Start of a new simulation (Batch or Single Shot)
            if re.match(r"(Batch simulation #|Single shot simulation #)", line, re.IGNORECASE):
                # Save previous record if it has the targets
                if 'PCE' in current_record and 'Voc' in current_record:
                    current_record['Source_File'] = os.path.basename(filepath)
                    dataset.append(current_record)
                current_record = {'V_list': [], 'J_list': []}
                in_batch_params = False
                in_data_table = False
                continue

            # Check for Temperature
            m_temp = re.match(r"^Temperature\s+([\d.eE+\-]+)\s+K", line)
            if m_temp:
                current_record['Temperature_K'] = float(m_temp.group(1))
            
            # Check for batch parameters section
            if "**Batch parameters**" in line:
                in_batch_params = True
                continue
                
            # If in batch parameters, extract them until we hit the data table
            if in_batch_params:
                if line == "" or "v(V)" in line:
                    in_batch_params = False
                else:
                    parts = line.split(":")
                    if len(parts) == 2:
                        param_name = parts[0].strip()
                        param_name = re.sub(r'[^a-zA-Z0-9_]', '_', param_name)
                        try:
                            param_val = float(parts[1].strip())
                            current_record[param_name] = param_val
                        except ValueError:
                            pass

            # Check for Data Table Start
            if "v(V)" in line and "jtot" in line:
                in_data_table = True
                continue
                
            if in_data_table:
                # Read data points
                if line == "" and len(current_record.get('V_list', [])) > 0:
                    # End of table
                    in_data_table = False
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            v_val = float(parts[0])
                            j_val = float(parts[1])
                            current_record['V_list'].append(v_val)
                            current_record['J_list'].append(j_val)
                        except ValueError:
                            pass

            # Extract target metrics
            m_voc = re.match(r"Voc\s*=\s*([\d.eE+\-]+)", line)
            m_jsc = re.match(r"Jsc\s*=\s*([\d.eE+\-]+)", line)
            m_ff  = re.match(r"FF\s*=\s*([\d.eE+\-]+)", line)
            m_eta = re.match(r"eta\s*=\s*([\d.eE+\-]+)", line)
            
            if m_voc: current_record['Voc'] = float(m_voc.group(1))
            if m_jsc: current_record['Jsc'] = float(m_jsc.group(1))
            if m_ff:  current_record['FF']  = float(m_ff.group(1))
            if m_eta: current_record['PCE'] = float(m_eta.group(1))
            
        # Append the last record if valid
        if 'PCE' in current_record and 'Voc' in current_record:
            current_record['Source_File'] = os.path.basename(filepath)
            dataset.append(current_record)
            
    return dataset

def main():
    print("Searching for .iv files in the workspace...")
    all_iv_files = glob.glob('../**/*.iv', recursive=True)
    if not all_iv_files:
        all_iv_files = glob.glob('**/*.iv', recursive=True)

    print(f"Found {len(all_iv_files)} .iv files. Extracting data...")
    
    all_data = []
    for filepath in all_iv_files:
        try:
            file_data = parse_iv_file(filepath)
            all_data.extend(file_data)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            
    if not all_data:
        print("No valid data extracted.")
        return
        
    df = pd.DataFrame(all_data)
    df.columns = [col.strip('_') for col in df.columns]
    
    # Save standard CSV without lists
    df_metrics = df.drop(columns=['V_list', 'J_list'], errors='ignore')
    output_csv = 'Machine_Learning_Dashboard/ml_dataset.csv'
    df_metrics.to_csv(output_csv, index=False)
    print(f"Saved metric dataset: {output_csv}")
    
    # Save Pickle with lists for PyTorch J-V curve training
    output_pkl = 'Machine_Learning_Dashboard/jv_curve_dataset.pkl'
    df.to_pickle(output_pkl)
    print(f"Saved J-V curve dataset: {output_pkl}")
    
if __name__ == "__main__":
    main()
