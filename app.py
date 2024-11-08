{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d1cbd741-143c-4a00-bdf0-05de7596331d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-11-07 19:44:42.850 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.859 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\ProgramData\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-11-07 19:44:43.859 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.859 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.859 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.859 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.887 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.891 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.900 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.903 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.908 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.908 Session state does not function when running a script without `streamlit run`\n",
      "2024-11-07 19:44:43.908 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.908 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.922 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.935 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.938 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.938 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.947 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.948 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.953 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.953 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.956 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.970 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.973 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.980 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.986 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.987 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.989 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.991 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.991 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:43.991 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:44.002 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:44.002 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:44.005 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:44.005 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-11-07 19:44:44.007 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Load the model\n",
    "with open('random_forest_telecom_churn.pkl', 'rb') as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "# Title and description\n",
    "st.title(\"Random Forest Model Deployment\")\n",
    "st.write(\"This app predicts the Iris species based on the sepal and petal measurements.\")\n",
    "\n",
    "# Input fields\n",
    "sepal_length = st.number_input(\"Sepal Length\", min_value=0.0, max_value=10.0, step=0.1)\n",
    "sepal_width = st.number_input(\"Sepal Width\", min_value=0.0, max_value=10.0, step=0.1)\n",
    "petal_length = st.number_input(\"Petal Length\", min_value=0.0, max_value=10.0, step=0.1)\n",
    "petal_width = st.number_input(\"Petal Width\", min_value=0.0, max_value=10.0, step=0.1)\n",
    "\n",
    "# Prediction button\n",
    "if st.button(\"Predict\"):\n",
    "    # Model expects a 2D array as input, reshape inputs accordingly\n",
    "    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])\n",
    "    prediction = model.predict(input_data)\n",
    "    st.write(f\"The predicted class is: {prediction[0]}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62f49c3e-ebb6-469a-8d8e-2b3b87158327",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
