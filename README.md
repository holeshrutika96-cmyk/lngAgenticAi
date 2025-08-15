LNG Transport Agentic System
Overview
The LNG Transport Agentic System is a Python-based application for optimizing LNG (liquefied natural gas) transport operations across multiple storage facilities, carriers, and cargos. It uses intelligent agents to monitor boil-off gas (BOG) rates, optimize ship routes, and schedule cargos based on storage forecasts. The system supports scenarios like heat_leak (increased BOG and temperature) and demand_spike (reduced storage levels due to high demand). Outputs include interactive Highcharts visualizations, CSV data exports, and detailed PDF reports (convertible to Word).
This README provides instructions to set up and run the system in C:\Aniket\AgenticAi\lngAgentic, execute the demand_spike and heat_leak scenarios, and generate/view reports.
Prerequisites

Operating System: Windows (tested on Windows 10/11)
Python: Version 3.8 or higher
LaTeX: TeX Live or MiKTeX for compiling report documents
Hardware: At least 8GB RAM, 4-core CPU recommended
Internet: Required for installing dependencies and optional MQTT data


1.Save the following files (provided separately):
lng_agent.py
lng_demand_spike_scenario.tex
lng_heat_leak_scenario.tex


2. Install Python Dependencies

Install Python 3.8+ from python.org if not already installed.
Create a virtual environment (optional but recommended):python -m venv venv
.\venv\Scripts\Activate.ps1


Install required Python packages:pip install numpy pandas langchain langchain_openai pulp paho-mqtt requests prophet cmdstanpy


Note: cmdstanpy requires a C++ compiler. If issues occur, follow CmdStanPy installation.
For prophet, ensure CmdStan is installed:python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"





3. Install LaTeX for Reports
To compile lng_demand_spike_scenario.tex and lng_heat_leak_scenario.tex into PDFs:

Install TeX Live (recommended) from tug.org/texlive or MiKTeX from miktex.org.
Ensure required LaTeX packages are installed:tlmgr install geometry booktabs pgfplots tikz noto


For MiKTeX, use the MiKTeX Console to install these packages.


Verify latexmk:latexmk --version



4. Install Pandoc (Optional for Word Conversion)
To convert PDFs to Word documents:

Install Pandoc from pandoc.org.
Verify:pandoc --version



5. Install Ollama for Local LLM
The script uses a local LLaMA 3 model via Ollama.

Download and install Ollama from ollama.com.
Pull the LLaMA 3 model:ollama pull llama3


Start the Ollama server:ollama serve



Project Structure

lng_agent.py: Main script for running the LNG transport system and scenarios.
lng_demand_spike_scenario.tex: LaTeX document for the demand spike scenario report.
lng_heat_leak_scenario.tex: LaTeX document for the heat leak scenario report.
log.txt: Generated log file for script execution.
lng_visualization_demand_spike.html: Interactive charts for the demand spike scenario.
lng_visualization_heat_leak.html: Interactive charts for the heat leak scenario.
lng_data_demand_spike.csv: Exported data for the demand spike scenario.
lng_data_heat_leak.csv: Exported data for the heat leak scenario.

Running the Project
1. Run the LNG Transport System

Navigate to the project directory:cd C:\Aniket\AgenticAi\lngAgentic


Run the script with the demand_spike scenario:python lng_agent.py > log.txt


This executes one iteration of the demand_spike scenario, producing:
lng_visualization_demand_spike.html: Interactive charts (BOG rate, temperature, storage levels, cargo schedules).
lng_data_demand_spike.csv: Exported data table.
log.txt: Execution logs.




To run the heat_leak scenario instead, modify lng_agent.py to set scenarios=['heat_leak'] in the __main__ block, then rerun:python lng_agent.py > log.txt



2. Compile LaTeX Reports

Compile the demand spike report:latexmk -pdf lng_demand_spike_scenario.tex


Output: lng_demand_spike_scenario.pdf


Compile the heat leak report:latexmk -pdf lng_heat_leak_scenario.tex


Output: lng_heat_leak_scenario.pdf


View PDFs using any PDF reader (e.g., Adobe Acrobat).

3. Convert PDFs to Word (Optional)
To convert the PDFs to Word documents:

Convert the demand spike report:pandoc lng_demand_spike_scenario.pdf -o lng_demand_spike_scenario.docx


Convert the heat leak report:pandoc lng_heat_leak_scenario.pdf -o lng_heat_leak_scenario.docx


Note: The tikz charts may appear static in Word. Refer to lng_visualization_*.html for interactive charts.



4. View Outputs

Interactive Charts:
Open lng_visualization_demand_spike.html or lng_visualization_heat_leak.html in a web browser (e.g., Chrome, Firefox) to view interactive Highcharts visualizations.


Data:
Open lng_data_demand_spike.csv or lng_data_heat_leak.csv in Excel or a text editor.


Logs:
Check log.txt for execution details, including agent decisions and errors.


Reports:
View lng_demand_spike_scenario.pdf or lng_heat_leak_scenario.pdf for detailed scenario analyses, including before/after data and charts.



Scenarios

Demand Spike:
Simulates a 500 m³ storage level drop per facility.
The Cargo Agent forecasts 90-day storage levels and schedules cargos (e.g., Cargo_1 to Carrier_1 for Storage_A) to prevent shortages.
Outputs: lng_visualization_demand_spike.html, lng_data_demand_spike.csv.


Heat Leak:
Simulates a 5°C temperature increase, 0.05%/day BOG rate increase, and 100 m³ storage drop.
The BOG Agent detects anomalies, the Route Agent adjusts speeds, and the Cargo Agent schedules deliveries.
Outputs: lng_visualization_heat_leak.html, lng_data_heat_leak.csv.



Troubleshooting

Python Script Errors:
Check log.txt for errors (e.g., missing dependencies, Ollama server not running).
Ensure Ollama server is running (ollama serve) and LLaMA 3 is pulled.
If prophet is slow, modify forecast_storage_level to use statsmodels (see script comments).
Run with one iteration for testing:python -c "from lng_agent import LNGTransportAgenticSystem; LNGTransportAgenticSystem().run_loop(iterations=1, scenarios=['demand_spike'])"




LaTeX Compilation Errors:
Ensure geometry, booktabs, pgfplots, tikz, and noto packages are installed.
Check lng_demand_spike_scenario.log or lng_heat_leak_scenario.log for errors.
Install missing packages:tlmgr install geometry booktabs pgfplots tikz noto




Pandoc Conversion:
If the Word document lacks formatting, try an online PDF-to-Word converter (e.g., Adobe Acrobat, SmallPDF).


MQTT Issues:
The script uses mock data by default. For real MQTT data, ensure connectivity to broker.hivemq.com:1883 and a valid topic (lng/tank/sensors).
Limit of 100 MQTT messages prevents overload.



Notes

Runtime: Each scenario iteration takes ~10 seconds (total ~30 seconds for 3 iterations). The script stops after 300 seconds or 5 agent errors to prevent infinite loops.
Data: Uses mock data (randomized within realistic ranges). Replace with real data via MQTT or CSV for production use.
Reports: The LaTeX documents include static tikz charts. For interactive charts, view the HTML outputs in a browser.
Customization:
Modify scenarios in lng_agent.py to test other scenarios (e.g., storm, shell_emission_audit).
Adjust thresholds in LNGTransportAgenticSystem.__init__ for stricter monitoring.



Contact
For issues or enhancements, contact the Shell LNG team or submit a GitHub issue (if hosted). Provide log.txt and any error messages for debugging.
