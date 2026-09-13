import streamlit as st
import os
import subprocess

def run_script(script_path):
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    return result.stdout, result.stderr

def main():
    st.title("Choose Detection Model")
    
    option = st.selectbox("Select a model", ("YOLO", "SSD"))

    if st.button("Run"):
        if option == "YOLO":
            with st.spinner("Running YOLO..."):
                os.chdir('YOLO')
                output, error = run_script('main.py')
                st.code(output)
                if error:
                    st.error(error)
                os.chdir('..')  # Return to the original directory

        elif option == "SSD":
            with st.spinner("Running SSD..."):
                os.chdir('SSD')
                output, error = run_script('social_distance.py')
                st.code(output)
                if error:
                    st.error(error)
                os.chdir('..')  # Return to the original directory

if __name__ == "__main__":
    main()
