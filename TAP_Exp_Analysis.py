import streamlit as st
from nptdms import TdmsFile
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="TAP Reactor Analysis",
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
    """Trapezoidal rule integration."""
    return np.trapezoid(y, x=x)

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
    Hp_val = F_peak / Mo_avg if Mo_avg > 0 else 0
    
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

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload .tdms File", type=["tdms"])
    
    st.divider()
    st.header("2. Global Settings")
    default_mom = 4.8674e-8
    moles_conv = st.number_input("Moles to Momentum Conv", value=default_mom, format="%.5e")

# --- Main Logic ---

if uploaded_file:
    with st.spinner("Processing TDMS file..."):
        tdms_file = TdmsFile.read(uploaded_file)
        
    meta_dict = parse_tdms_metadata(tdms_file)
    
    # Get available AMU groups
    all_groups = [g.name for g in tdms_file.groups()]
    data_groups = [g for g in all_groups if g not in ['Meta Data', 'Secondary Data']]
    
    # 1. SELECT AMU
    col_sel, col_info = st.columns([1, 3])
    with col_sel:
        idx = data_groups.index('2') if '2' in data_groups else 0
        selected_group = st.selectbox("Select AMU Group", data_groups, index=idx)
    
    if selected_group:
        # 2. CALIBRATION FACTOR (Dependent on AMU)
        st.markdown("---")
        col_cal1, col_cal2 = st.columns([1, 3])
        
        with col_cal1:
            # MOVED HERE per request
            default_cal = 0.557839
            calibration_factor = st.number_input(
                f"Calibration Factor (AMU {selected_group})", 
                value=default_cal, 
                format="%.6f"
            )
        
        # Calculate Effective Conversion
        conversion_factor = calibration_factor * moles_conv
        
        with col_cal2:
            st.info(f"**Effective Conversion Factor:** `{conversion_factor:.4e}` (Flux = Intensity × Cal × MomConv)")

        # 3. DATA PROCESSING
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

            # Preprocess: Slice (remove edges) & Convert
            df_proc = df_raw.iloc[1:-1].copy()
            df_proc[time_col_name] = pd.to_numeric(df_proc[time_col_name], errors='coerce')
            
            # Apply Conversion
            for c in pulse_cols:
                df_proc[c] = pd.to_numeric(df_proc[c], errors='coerce') * conversion_factor
            
            # --- TABS ---
            st.divider()
            tab1, tab2, tab3 = st.tabs(["📈 Interactive Plots", "📊 Analysis Summary", "💾 Metadata & Export"])

            with tab1:
                st.subheader(f"Pulse Responses (Mass {meta_dict.get(f'AMU {selected_group}', 'Unknown')})")
                
                fig = go.Figure()
                pulses_to_show = st.multiselect("Select Pulses to Visualize", pulse_cols, default=pulse_cols)

                for pulse in pulses_to_show:
                    fig.add_trace(go.Scatter(
                        x=df_proc[time_col_name],
                        y=df_proc[pulse],
                        mode='lines',
                        name=f"Pulse {pulse}",
                        line=dict(width=1.5)
                    ))

                fig.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Exit Flow (nmol/s)",
                    template="plotly_white",
                    hovermode="x unified",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # --- Multi-Peak Configuration ---
                st.subheader("Peak Detection Settings")
                
                col_cfg1, col_cfg2 = st.columns([1, 3])
                with col_cfg1:
                    n_peaks = st.number_input("Number of Peaks per Pulse", min_value=1, value=1, step=1)
                
                sorted_delays = []
                with col_cfg2:
                    st.write("**Specify Start/Delay Times (s) in chronological order:**")
                    cols = st.columns(n_peaks)
                    for i in range(n_peaks):
                        with cols[i]:
                            def_val = 0.1 if i == 0 else 0.1 + (i * 1.0)
                            val = st.number_input(f"Delay {i+1}", value=def_val, step=0.1, format="%.2f", key=f"d_{i}")
                            sorted_delays.append(val)
                
                st.divider()
                
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
                st.subheader("Full Preprocessed Data (Flux)")
                st.dataframe(df_proc, use_container_width=True)
                
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    csv_full = df_proc.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Processed Data (CSV)",
                        csv_full,
                        f"tap_full_data_amu_{selected_group}.csv",
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
                        label="📥 Download Full Data (Excel .xlsx)",
                        data=buffer,
                        file_name=f"tap_data_amu_{selected_group}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

else:
    st.info("👋 Please upload a .tdms file to begin.")