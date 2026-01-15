import streamlit as st
from nptdms import TdmsFile
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# Set Pandas option to allow rendering large styled dataframes
# The error reported was for ~2.3M cells, so we set a safe limit of 5M.
pd.set_option("styler.render.max_elements", 5000000)

# --- Page Configuration ---
st.set_page_config(
    page_title="TAP Reactor Experimental Analysis",
    page_icon="🧪",
    layout="wide"
)

# --- Helper Functions ---

def parse_tdms_metadata(tdms_file):
    """Extracts key-value pairs from the 'Meta Data' group."""
    meta_dict = {}
    try:
        if 'Meta Data' in tdms_file:
            group = tdms_file['Meta Data']
            if 'Item' in group and 'Value' in group:
                items = group['Item'][:]
                values = group['Value'][:]
                for i, v in zip(items, values):
                    key = i.decode('utf-8') if isinstance(i, bytes) else str(i)
                    meta_dict[key] = v
    except Exception as e:
        st.warning(f"Metadata warning: {e}")
    
    for key, val in tdms_file.properties.items():
        meta_dict[key] = val
    return meta_dict

def get_group_data(tdms_file, group_name):
    """Converts a specific TDMS group into a DataFrame."""
    try:
        group = tdms_file[group_name]
    except KeyError:
        return None, f"Group {group_name} not found."

    data = {}
    max_len = 0
    for channel in group.channels():
        max_len = max(max_len, len(channel))

    for channel in group.channels():
        vals = channel[:]
        if len(vals) < max_len:
            vals = np.pad(vals, (0, max_len - len(vals)), constant_values=np.nan)
        data[channel.name] = vals

    df = pd.DataFrame(data)
    return df, None

# --- CORE LOGIC (Based on Notebook) ---

def get_t_p(time_vector, response_vector, t_delay=0):
    """Calculates peak time (tp) relative to delay."""
    if len(time_vector) != len(response_vector):
        return np.nan
    max_index = np.argmax(response_vector)
    return time_vector[max_index] - t_delay

def area_under_curve(x, y):
    """Numerical integration of the provided signal."""
    # Use np.trapezoid (modern replacement for np.trapz)
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x)
    else:
        return np.trapz(y, x=x)

def validate_knudsen_criteria(t, F_exit_segment, delay_time=0):
    """
    Calculates Knudsen criteria parameters for a specific data segment.
    Ref: User Jupyter Notebook
    """
    t_list = np.array(t)
    f_list = np.array(F_exit_segment)
    
    if len(f_list) == 0:
        return {'tp': 0, 'Hp': 0, 'Hp_tp': 0, 'M0': 0, 'Status': "No Data", 'F_peak': 0}

    # 1. Calculate Peak Time (relative)
    t_p_val = get_t_p(t_list, f_list, delay_time)
    
    # 2. Area under the curve (M0)
    Mo_avg = area_under_curve(t_list, f_list)
    
    # 3. Peak max value (Raw Flux)
    F_peak = np.max(f_list)
    
    # 4. Peak Height (Hp) normalized by Area (M0) -> Units: 1/s
    Hp_val = F_peak / Mo_avg if Mo_avg != 0 else 0
    
    # 5. Knudsen Product (Dimensionless)
    knudsen_product = Hp_val * t_p_val
    
    # 6. Status Check
    status = "Check"
    if 0.25 <= knudsen_product <= 0.35:
        status = "✓ EXCELLENT"
    
    return {
        'tp': t_p_val, 
        'Hp': Hp_val, 
        'Hp_tp': knudsen_product, 
        'M0': Mo_avg,
        'Status': status,
        'F_peak': F_peak
    }

# --- Main App Interface ---

st.title("TAP Reactor Experiment Analysis")

# File Upload on Main Page
uploaded_file = st.file_uploader("Upload .tdms File", type=["tdms"])

# --- Main Logic ---

if uploaded_file:
    with st.spinner("Processing TDMS file..."):
        tdms_file = TdmsFile.read(uploaded_file)
        
    meta_dict = parse_tdms_metadata(tdms_file)
    
    # Get available AMU groups
    all_groups = [g.name for g in tdms_file.groups()]
    data_groups = [g for g in all_groups if g not in ['Meta Data', 'Secondary Data']]
    
    # --- Session State Reset on New File ---
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if "current_file_id" not in st.session_state or st.session_state.current_file_id != file_id:
        st.session_state.current_file_id = file_id
        # Clear all peak-related and calibration/baseline dynamic keys
        for k in list(st.session_state.keys()):
            if any(k.startswith(prefix) for prefix in ["d_global_", "c_global_", "cal_amu_", "bs_start_", "bs_end_"]) or k == "n_peaks_val":
                del st.session_state[k]
    
    
    # 1. SELECT AMU
    col_sel, col_info = st.columns([1, 3])
    with col_sel:
        idx = data_groups.index('1') if '1' in data_groups else 0
        selected_group = st.selectbox("Select AMU Group", data_groups, index=idx)
    
    if selected_group:
        # 2. CALIBRATION FACTOR (Dependent on AMU)
        st.markdown("---")
        col_cal1, col_cal2 = st.columns([1, 3])
        
        with col_cal1:
            default_cal = 0.557839
            cal_key = f"cal_amu_{selected_group}"
            if cal_key not in st.session_state:
                st.session_state[cal_key] = default_cal
            
            calibration_factor = st.number_input(
                f"Calibration Factor (AMU {selected_group})", 
                format="%.6f",
                key=cal_key
            )
        
        with col_cal2:
            st.info(f"**Calibration Multiplier:** `{calibration_factor:.4f}` for AMU {selected_group}. Pulse-specific Intensity -> Flux conversion factors will be multiplied by this value.")

        # 3. DATA PROCESSING INITIALIZATION
        df_raw, error = get_group_data(tdms_file, selected_group)
        
        if error:
            st.error(error)
        else:
            time_cols = [c for c in df_raw.columns if "Time" in c]
            if not time_cols:
                st.error("No 'Time' column found.")
                st.stop()
            
            time_col_name = time_cols[0]
            pulse_cols = [c for c in df_raw.columns if c not in ['Item', 'Value', time_col_name]]
            
            try:
                pulse_cols = sorted(pulse_cols, key=lambda x: float(x))
            except:
                pass

            # Preprocess: Slice (remove edges)
            df_proc_raw = df_raw.iloc[1:-1].copy()
            df_proc_raw[time_col_name] = pd.to_numeric(df_proc_raw[time_col_name], errors='coerce')
            for c in pulse_cols:
                df_proc_raw[c] = pd.to_numeric(df_proc_raw[c], errors='coerce')

            # --- BASELINE CORRECTION SETTINGS (MOVED UP) ---
            st.markdown("---")
            st.subheader("Baseline Correction Settings")
            
            correction_method = st.selectbox(
                "Correction Method",
                ["No correction", "Time range average", "Absolute AMU minimum", "Custom correction value"],
                index=1,
                help="Select how to correct the Y-axis baseline *before* conversion to flux."
            )
            
            # Use raw data for baseline determination
            raw_intensity_data = df_proc_raw[pulse_cols].values.flatten()
            baseline_offset_raw = 0.0
            
            if correction_method == "No correction":
                baseline_offset_raw = 0.0
                apply_baseline = False
                st.markdown("Using **raw intensity** as is.")

            elif correction_method == "Absolute AMU minimum":
                baseline_offset_raw = np.min(raw_intensity_data)
                apply_baseline = True
                st.success(f"Subtracting Raw Intensity Minimum: `{baseline_offset_raw:.4e}`")

            elif correction_method == "Time range average":
                t_min = float(df_proc_raw[time_col_name].min())
                t_max = float(df_proc_raw[time_col_name].max())
                
                c_bl1, c_bl2 = st.columns(2)
                def_start = max(t_min, 0.01)
                def_end = min(t_max, 0.09)
                
                bs_start_key = f"bs_start_{selected_group}"
                bs_end_key = f"bs_end_{selected_group}"
                
                if bs_start_key not in st.session_state:
                    st.session_state[bs_start_key] = float(def_start)
                if bs_end_key not in st.session_state:
                    st.session_state[bs_end_key] = float(def_end)

                t_start_avg = c_bl1.number_input("Avg Start Time (s)", min_value=t_min, max_value=t_max, step=0.01, format="%.3f", key=bs_start_key)
                t_end_avg = c_bl2.number_input("Avg End Time (s)", min_value=t_min, max_value=t_max, step=0.01, format="%.3f", key=bs_end_key)
                
                mask_map = (df_proc_raw[time_col_name] >= t_start_avg) & (df_proc_raw[time_col_name] <= t_end_avg)
                
                if mask_map.any():
                    subset_data = df_proc_raw.loc[mask_map, pulse_cols].values.flatten()
                    baseline_offset_raw = np.mean(subset_data)
                    apply_baseline = True
                    st.success(f"Subtracting Average Raw Intensity ({t_start_avg}-{t_end_avg}s): `{baseline_offset_raw:.4e}`")
                else:
                    st.warning("No data found in specified time range.")
                    apply_baseline = False

            elif correction_method == "Custom correction value":
                custom_val_key = f"custom_bl_{selected_group}"
                if custom_val_key not in st.session_state:
                    st.session_state[custom_val_key] = -0.03
                
                baseline_offset_raw = st.number_input("Custom Raw Intensity Offset", step=0.01, format="%.4f", key=custom_val_key)
                apply_baseline = True
                st.success(f"Subtracting Custom Raw Value: `{baseline_offset_raw}`")

            # --- PEAK DETECTION SETTINGS ---
            st.markdown("---")
            st.subheader("Peak Detection Settings")
            
            # Extract delays from metadata
            meta_delays = []
            for i in range(1, 5):
                key = f"Delay Time {i}"
                val = meta_dict.get(key, 0)
                try:
                    val_float = float(val)
                    if val_float > 0:
                        meta_delays.append(val_float)
                except (ValueError, TypeError):
                    continue
            
            def_n_peaks = len(meta_delays) if meta_delays else 1
            
            col_cfg1, col_cfg2 = st.columns([1, 3])
            with col_cfg1:
                if "n_peaks_val" not in st.session_state:
                    st.session_state.n_peaks_val = def_n_peaks
                n_peaks = st.number_input("Number of Peaks per Pulse", min_value=1, step=1, key="n_peaks_val")
                default_mom = 4.8674e-8
            
            peak_configs = []
            with col_cfg2:
                st.write("**Pulse Start Times (s) & Conversion Factors (nmol/s per unit intensity):**")
                for i in range(n_peaks):
                    if i < len(meta_delays):
                        def_d = float(meta_delays[i])
                    else:
                        def_d = float(0.1 if i == 0 else 0.1 + (i * 1.0))
                    
                    d_key = f"d_global_{i}"
                    c_key = f"c_global_{i}"
                    
                    if d_key not in st.session_state:
                        st.session_state[d_key] = def_d
                    if c_key not in st.session_state:
                        st.session_state[c_key] = default_mom
                    
                    c_p1, c_p2 = st.columns(2)
                    with c_p1:
                        d_val = st.number_input(f"Delay {i+1}", step=0.1, format="%.2f", key=d_key)
                    with c_p2:
                        c_val = st.number_input(f"Intensity -> Flux Conversion {i+1}", format="%.5e", key=c_key)
                    
                    peak_configs.append({'delay': d_val, 'conv': c_val})
            
            peak_configs.sort(key=lambda x: x['delay'])
            sorted_delays = [p['delay'] for p in peak_configs]

            # --- APPLY CORRECTION AND CONVERSION ---
            df_proc = df_proc_raw.copy()
            
            for c in pulse_cols:
                # 1. Apply Baseline Correction to RAW
                raw_vals = df_proc_raw[c].values
                corrected_raw = raw_vals - baseline_offset_raw if apply_baseline else raw_vals
                
                # 2. Apply Piecewise Conversion to CORRECTED RAW
                flux_column = np.zeros_like(corrected_raw)
                times = df_proc_raw[time_col_name].values
                
                for i, config in enumerate(peak_configs):
                    # Start of segment:
                    # For the first peak, start from the very beginning of the data (times[0])
                    # so that baseline noise before the peak is converted and visible.
                    t_start = times[0] if i == 0 else config['delay']
                    
                    if i < len(peak_configs) - 1:
                        t_end = peak_configs[i+1]['delay']
                    else:
                        t_end = times[-1] + 0.1
                    
                    mask = (times >= t_start) & (times < t_end)
                    eff_conv = config['conv'] * calibration_factor
                    flux_column[mask] = corrected_raw[mask] * eff_conv
                
                df_proc[c] = flux_column
            
            # --- TABS ---
            tab1, tab2, tab3 = st.tabs(["📈 Interactive Plots", "📊 Analysis Summary", "💾 Metadata & Export"])

            with tab1:
                st.subheader(f"Pulse Response Flux Profiles (Mass {meta_dict.get(f'AMU {selected_group}', 'Unknown')})")
                
                # Pulse Selection
                pulses_to_show = st.multiselect("Select Pulses to Visualize", pulse_cols, default=pulse_cols)
                
                fig = go.Figure()
                for pulse in pulses_to_show:
                    y_data = df_proc[pulse]
                    
                    fig.add_trace(go.Scatter(
                        x=df_proc[time_col_name],
                        y=y_data,
                        mode='lines',
                        name=f"Pulse {pulse}",
                        line=dict(width=1.5)
                    ))

                fig.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Exit Flow (nmol/s)" + (" [corrected]" if apply_baseline else ""),
                    template="plotly_white",
                    hovermode="x unified",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- RAW SIGNAL PLOT & EXPORT ---
                st.divider()
                st.subheader(f"Raw MS Signal (Mass {meta_dict.get(f'AMU {selected_group}', 'Unknown')})")
                
                fig_raw = go.Figure()
                for pulse in pulses_to_show:
                    # Raw data from processing (no conversion factor)
                    y_raw = df_proc_raw[pulse]
                    
                    fig_raw.add_trace(go.Scatter(
                        x=df_proc_raw[time_col_name],
                        y=y_raw,
                        mode='lines',
                        name=f"Pulse {pulse} (Raw)",
                        line=dict(width=1.5) 
                    ))

                fig_raw.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Raw Signal (Intensity)",
                    template="plotly_white",
                    hovermode="x unified",
                    height=500
                )
                st.plotly_chart(fig_raw, use_container_width=True)
                
                # Excel Export logic
                if st.button("Prepare Raw Data for Download"):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_proc_raw.to_excel(writer, index=False, sheet_name=f'Raw_AMU_{selected_group}')
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Raw Data (.xlsx)",
                        data=excel_data,
                        file_name=f"TAP_Raw_AMU_{meta_dict.get(f'AMU {selected_group}', 'Unknown')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            with tab2:
                # --- Analysis Loop ---
                results = []
                time_vector = df_proc[time_col_name].values
                sorted_delays.sort()
                
                for p_idx, pulse in enumerate(pulse_cols):
                    current_pulse_data = df_proc[pulse].values
                    
                    for i, start_delay in enumerate(sorted_delays):
                        t_start = start_delay
                        # End is next delay OR end of data
                        if i < len(sorted_delays) - 1:
                            t_end = sorted_delays[i+1]
                        else:
                            t_end = time_vector[-1]
                        
                        mask = (time_vector >= t_start) & (time_vector <= t_end)
                        t_segment = time_vector[mask]
                        y_segment = current_pulse_data[mask]
                        
                        stats = validate_knudsen_criteria(t_segment, y_segment, delay_time=start_delay)
                        
                        stats['Pulse'] = pulse
                        stats['Peak_Num'] = i + 1
                        stats['Delay'] = start_delay
                        
                        results.append(stats)
                
                df_results = pd.DataFrame(results)
                
                # Column ordering
                desired_order = ['Pulse', 'Peak_Num', 'Delay', 'M0', 'tp', 'Hp', 'Hp_tp', 'Status']
                cols_to_use = [c for c in desired_order if c in df_results.columns]
                df_display = df_results[cols_to_use]
                
                st.subheader("Analysis Results")
                st.markdown("""
                **Legend:**
                * **M0:** Total Area (mol)
                * **tp:** Peak Time (s) [Relative to Delay]
                * **Hp:** Normalized Peak Height ($1/s$) calculated as $F_{peak} / M_0$
                * **Hp_tp:** Knudsen Product ($H_p \\times t_p$) - Target: 0.25 - 0.35
                """)
                
                format_dict = {
                    "Delay": "{:.2f}",
                    "M0": "{:.4e}",
                    "tp": "{:.4f}",
                    "Hp": "{:.4f}",
                    "Hp_tp": "{:.4f}"
                }
                
                st.dataframe(df_display.style.format(format_dict), use_container_width=True)
                
                csv_summary = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Summary CSV",
                    csv_summary,
                    "tap_analysis_knudsen.csv",
                    "text/csv"
                )

            with tab3:
                st.subheader("File Metadata & Export")
                
                with st.expander("Show Metadata"):
                    if meta_dict:
                        meta_df = pd.DataFrame(list(meta_dict.items()), columns=["Item", "Value"])
                        st.dataframe(meta_df, hide_index=True)
                    else:
                        st.info("No metadata found.")

                st.divider()
                st.subheader("Full Processed Data (Flux)")
                
                # Format pulse columns to display scientific notation to avoid showing zeros
                format_columns = {col: "{:.4e}" for col in pulse_cols}
                st.dataframe(df_proc.style.format(format_columns), use_container_width=True)
                
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    csv_full = df_proc.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Processed Data (CSV)",
                        csv_full,
                        f"tap_full_data_amu_{meta_dict.get(f'AMU {selected_group}', 'Unknown')}.csv",
                        "text/csv",
                        key='download_full_csv'
                    )
                
                with col_d2:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_proc.to_excel(writer, sheet_name='Processed_Data', index=False)
                        df_display.to_excel(writer, sheet_name='Analysis_Results', index=False)
                        if meta_dict:
                            pd.DataFrame(list(meta_dict.items()), columns=["Item", "Value"]).to_excel(writer, sheet_name="Metadata", index=False)
                    
                    st.download_button(
                        label="📥 Download Processed Data (Excel .xlsx)",
                        data=buffer,
                        file_name=f"tap_data_amu_{meta_dict.get(f'AMU {selected_group}', 'Unknown')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

else:
    st.info("📤 Please upload a .tdms file to begin.")