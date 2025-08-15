import numpy as np
import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import paho.mqtt.client as mqtt
import requests
from prophet import Prophet
import time
import json
from typing import Dict, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cmdstanpy as Prophet backend
import cmdstanpy
cmdstanpy.install_cmdstan()  # Install CmdStan if not already installed

# Custom timeout decorator for Windows compatibility
def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except TimeoutError:
                    logger.error(f"{func.__name__} timed out after {seconds} seconds")
                    raise TimeoutError(f"{func.__name__} timed out")
        return wrapper
    return decorator

# Initialize Ollama LLM via OpenAI-compatible client
llm = OpenAI(
    api_key="ac8a79319c25473788cc1eb3287a2dd5.UzVrs3Et_V_98Sxnxx39Jlw-",  # Empty for local Ollama; set if using hosted instance
    base_url="http://localhost:11434/v1",
    model="gemma3:1b"  # Adjust based on your Ollama model
)

# Tools for Agents
@tool
def get_weather_forecast(latitude: float, longitude: float) -> str:
    """Fetch weather data (mock; replace with OpenWeatherMap API)."""
    logger.info(f"Calling get_weather_forecast with lat={latitude}, lon={longitude}")
    if np.random.rand() > 0.7:
        return f"Storm warning at ({latitude}, {longitude})."
    return f"Clear weather at ({latitude}, {longitude})."

@tool
def calculate_bog_rate(temperature: float, pressure: float) -> float:
    """Simulate thermodynamic BOG calculation (replace with CoolProp)."""
    logger.info(f"Calling calculate_bog_rate with temp={temperature}, pressure={pressure}")
    base_bog = 0.1
    temp_effect = (temperature + 162) * 0.01
    return base_bog + temp_effect

@tool
def calculate_emissions(bog_rate: float, distance: float) -> float:
    """Calculate methane/CO2 emissions for Shell compliance (mock)."""
    logger.info(f"Calling calculate_emissions with bog_rate={bog_rate}, distance={distance}")
    try:
        bog_rate = float(bog_rate)
        distance = float(distance)
        methane_slip = bog_rate * 0.05
        co2 = distance * 0.1
        return methane_slip + co2
    except Exception as e:
        logger.error(f"Error in calculate_emissions: {e}")
        raise

@tool
def forecast_storage_level(tool_input: Dict[str, Any] = None, historical_data: str = None, storage_id: str = None) -> str:
    """Forecast LNG storage levels for a specific storage facility using Prophet with cmdstanpy (returns JSON string)."""
    logger.info(f"Calling forecast_storage_level with tool_input={tool_input}, historical_data={historical_data}, storage_id={storage_id}")
    try:
        # Handle LangChain tool_input (single dict) or direct arguments
        if tool_input is not None and isinstance(tool_input, dict):
            historical_data = tool_input.get('historical_data')
            storage_id = tool_input.get('storage_id')
        if not historical_data or not storage_id:
            raise ValueError("Both historical_data and storage_id are required")
        
        data = pd.read_json(historical_data)
        df = data[data['storage_id'] == storage_id][['time', 'storage_level']].rename(columns={'time': 'ds', 'storage_level': 'y'})
        if df.empty:
            logger.warning(f"No data for storage_id={storage_id}, returning empty forecast")
            return pd.DataFrame().to_json()
        df['ds'] = pd.to_datetime(df['ds'], unit='s')
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=90, freq='D')  # 3-month forecast
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']].tail(90).to_json()
    except Exception as e:
        logger.error(f"Error in forecast_storage_level for storage_id={storage_id}: {e}")
        raise

@tool
def optimize_cargo_schedule(demands: str, storage_forecasts: str, carriers: str) -> str:
    """Optimize LNG cargo schedules across multiple carriers and storage facilities (returns JSON string)."""
    logger.info(f"Calling optimize_cargo_schedule with demands={demands}")
    try:
        demands = json.loads(demands)
        storage_forecasts = json.loads(storage_forecasts)
        carriers = json.loads(carriers)
        prob = LpProblem("Cargo_Scheduling", LpMinimize)
        assignments = LpVariable.dicts("Assign", [(c['id'], d['id']) for c in carriers for d in demands], cat='Binary')
        prob += lpSum(assignments[(c['id'], d['id'])] for c in carriers for d in demands), "Minimize_Assignments"
        
        # Demand constraints
        for d in demands:
            prob += lpSum(assignments[(c['id'], d['id'])] * c['capacity'] for c in carriers) >= d['volume'], f"Demand_{d['id']}"
        
        # Carrier capacity constraints
        for c in carriers:
            prob += lpSum(assignments[(c['id'], d['id'])] * d['volume'] for d in demands) <= c['capacity'], f"Capacity_{c['id']}"
        
        # Storage constraints
        for storage_id, forecast in storage_forecasts.items():
            forecast_df = pd.read_json(forecast)
            if not forecast_df.empty:
                prob += lpSum(assignments[(c['id'], d['id'])] * d['volume'] for c in carriers for d in demands if d['storage_id'] == storage_id) <= forecast_df['yhat'].mean(), f"Storage_{storage_id}"
        
        prob.solve()
        schedule = [
            {"carrier_id": c['id'], "cargo_id": d['id'], "storage_id": d['storage_id'], "assign": bool(assignments[(c['id'], d['id'])].value())}
            for c in carriers for d in demands
        ]
        return json.dumps(schedule)
    except Exception as e:
        logger.error(f"Error in optimize_cargo_schedule: {e}")
        raise

class LNGTransportAgenticSystem:
    def __init__(self):
        # Initialize state
        self.lng_data = pd.DataFrame(columns=['time', 'storage_id', 'temperature', 'pressure', 'bog_rate', 'speed', 'distance', 'latitude', 'longitude', 'emissions', 'storage_level'])
        self.storage_facilities = ['Storage_A', 'Storage_B', 'Storage_C']
        self.carriers = [
            {'id': 'Carrier_1', 'capacity': 3000},
            {'id': 'Carrier_2', 'capacity': 4000},
            {'id': 'Carrier_3', 'capacity': 2500}
        ]
        self.cargos = [
            {'id': 'Cargo_1', 'volume': 3000, 'storage_id': 'Storage_A'},
            {'id': 'Cargo_2', 'volume': 5000, 'storage_id': 'Storage_B'},
            {'id': 'Cargo_3', 'volume': 2000, 'storage_id': 'Storage_C'}
        ]
        self.thresholds = {
            'bog_rate_max': 0.15,
            'temperature_max': -160,
            'route_fuel_min': 1000,
            'emissions_max': 50.0,
            'storage_min': 1000,
        }
        self.alerts = []
        self.shared_context = {"latest_data": {}, "actions": [], "forecasts": {}, "cargo_schedule": None}
        self.mqtt_client = None
        self.mqtt_message_count = 0
        self.max_mqtt_messages = 100
        self.agent_errors = 0
        self.max_agent_errors = 5
        
        # Define tools
        self.tools = [get_weather_forecast, calculate_bog_rate, calculate_emissions, forecast_storage_level, optimize_cargo_schedule]
        self.tool_names = ", ".join([tool.name for tool in self.tools])
        self.tools_description = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # LangChain agents
        bog_prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            template="""You are a BOG Monitoring Agent for an LNG tanker. Minimize BOG losses across multiple storage facilities. Use tools to check temperature, pressure, BOG rate, emissions, and storage levels per facility. For calculate_emissions, pass inputs as {"bog_rate": value, "distance": value}. If anomalies (BOG > 0.15%/day, temp > -160C, emissions > 50, storage < 1000 m3), mitigate and notify Route and Cargo Agents.
            
Available tools:
{tools}

Tool names: {tool_names}

Input: {input}
Scratchpad: {agent_scratchpad}"""
        )
        route_prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            template="""You are a Route Optimization Agent for multiple LNG carriers. Minimize fuel use and ensure timely delivery. Use weather forecasts, speed/distance data, emissions, and storage forecasts from tools. For calculate_emissions, pass inputs as {"bog_rate": value, "distance": value}. Collaborate with BOG and Cargo Agents.
            
Available tools:
{tools}

Tool names: {tool_names}

Input: {input}
Scratchpad: {agent_scratchpad}"""
        )
        cargo_prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            template="""You are a Cargo Scheduling Agent for multiple LNG cargos and carriers. Optimize schedules based on storage forecasts, cargo demands, and carrier availability across multiple storage facilities. Use forecast_storage_level with a single dictionary input {'historical_data': JSON string, 'storage_id': string}, and optimize_cargo_schedule tools. Collaborate with Route and BOG Agents.
            
Available tools:
{tools}

Tool names: {tool_names}

Input: {input}
Scratchpad: {agent_scratchpad}"""
        )
        self.bog_agent = create_react_agent(llm, self.tools, bog_prompt)
        self.route_agent = create_react_agent(llm, self.tools, route_prompt)
        self.cargo_agent = create_react_agent(llm, self.tools, cargo_prompt)
        self.bog_executor = AgentExecutor(agent=self.bog_agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        self.route_executor = AgentExecutor(agent=self.route_agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        self.cargo_executor = AgentExecutor(agent=self.cargo_agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        
        # MQTT setup
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_message = self.on_mqtt_message
            self.mqtt_client.connect("broker.hivemq.com", 1883, 60)
            self.mqtt_client.subscribe("lng/tank/sensors")
            self.mqtt_client.loop_start()
            logger.info("MQTT client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MQTT client: {e}")
            self.mqtt_client = None

    def on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT sensor data with message limit."""
        if self.mqtt_message_count >= self.max_mqtt_messages:
            logger.warning("Max MQTT messages reached, stopping MQTT loop")
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            return
        try:
            data = json.loads(msg.payload.decode())
            storage_id = data.get('storage_id', np.random.choice(self.storage_facilities))
            new_data = pd.DataFrame([{
                'time': time.time(),
                'storage_id': storage_id,
                'temperature': float(data.get('temperature', -162)),
                'pressure': float(data.get('pressure', 1.1)),
                'bog_rate': float(data.get('bog_rate', 0.1)),
                'speed': np.nan,
                'distance': np.nan,
                'latitude': np.nan,
                'longitude': np.nan,
                'emissions': np.nan,
                'storage_level': float(data.get('storage_level', 5000))
            }])
            self.lng_data = pd.concat([self.lng_data, new_data], ignore_index=True)
            self.shared_context["latest_data"][storage_id] = new_data
            self.mqtt_message_count += 1
            logger.info(f"Received MQTT data for storage_id={storage_id} (count: {self.mqtt_message_count}/{self.max_mqtt_messages})")
        except json.JSONDecodeError:
            logger.error("Invalid MQTT message format")

    def collect_ais_data(self, carrier_id: str):
        """Fetch ship data via AIS API for a specific carrier (mock)."""
        logger.info(f"Collecting AIS data for carrier_id={carrier_id}")
        return {
            'speed': float(np.random.normal(18, 2)),
            'distance': float(np.random.normal(5000, 500)),
            'latitude': 25.0,
            'longitude': -80.0
        }

    def initialize_default_data(self):
        """Initialize default data for each storage facility."""
        logger.info("Initializing default data")
        for storage_id in self.storage_facilities:
            default_data = pd.DataFrame([{
                'time': time.time(),
                'storage_id': storage_id,
                'temperature': float(np.random.normal(-162, 1)),
                'pressure': float(np.random.normal(1.1, 0.1)),
                'bog_rate': float(np.random.normal(0.1, 0.02)),
                'speed': float(np.random.normal(18, 2)),
                'distance': float(np.random.normal(5000, 500)),
                'latitude': 25.0,
                'longitude': -80.0,
                'emissions': np.nan,
                'storage_level': float(np.random.normal(5000, 500))
            }])
            self.lng_data = pd.concat([self.lng_data, default_data], ignore_index=True)
            self.shared_context["latest_data"][storage_id] = default_data
        logger.info(f"Initialized data: {self.lng_data.shape}")
        return self.lng_data

    def simulate_scenario(self, scenario_type: str, data: pd.DataFrame):
        """Simulate scenarios for each storage facility."""
        logger.info(f"Simulating scenario: {scenario_type}")
        if data.empty:
            logger.warning("DataFrame is empty, initializing default data")
            data = self.initialize_default_data()
        
        new_data = []
        for storage_id in self.storage_facilities:
            latest = data[data['storage_id'] == storage_id].iloc[-1].copy() if not data[data['storage_id'] == storage_id].empty else data.iloc[-1].copy()
            latest['storage_id'] = storage_id
            if scenario_type == 'heat_leak':
                latest['temperature'] += 5
                latest['bog_rate'] += 0.05
                latest['storage_level'] -= 100
                latest['emissions'] = calculate_emissions.run(
                    {"bog_rate": float(latest['bog_rate']), "distance": float(latest['distance'])}
                )
            elif scenario_type == 'storm':
                latest['speed'] *= 0.8
                latest['emissions'] = calculate_emissions.run(
                    {"bog_rate": float(latest['bog_rate']), "distance": float(latest['distance'])}
                ) * 1.2
            elif scenario_type == 'shell_emission_audit':
                latest['emissions'] = calculate_emissions.run(
                    {"bog_rate": float(latest['bog_rate']), "distance": float(latest['distance'])}
                ) + 10
                if latest['emissions'] > self.thresholds['emissions_max']:
                    self.alerts.append(f"Shell emission threshold exceeded for {storage_id}")
            elif scenario_type == 'demand_spike':
                latest['storage_level'] -= 500
            else:
                logger.warning(f"Unknown scenario: {scenario_type}")
            new_data.append(latest)
        
        self.lng_data = pd.concat([self.lng_data, pd.DataFrame(new_data)], ignore_index=True)
        logger.info(f"Simulated {scenario_type} scenario: {len(new_data)} records")
        return self.lng_data

    def collect_data(self):
        """Collect real-time data for all carriers and storage facilities."""
        logger.info("Collecting data")
        new_data = []
        for storage_id in self.storage_facilities:
            ais_data = self.collect_ais_data(np.random.choice([c['id'] for c in self.carriers]))
            latest_data = self.shared_context["latest_data"].get(storage_id)
            if latest_data is None or latest_data.empty:
                logger.warning(f"No latest data for storage_id={storage_id}, initializing default")
                self.initialize_default_data()
                latest_data = self.shared_context["latest_data"][storage_id]
            
            latest_data['speed'] = ais_data['speed']
            latest_data['distance'] = ais_data['distance']
            latest_data['latitude'] = ais_data['latitude']
            latest_data['longitude'] = ais_data['longitude']
            latest_data['emissions'] = calculate_emissions.run(
                {"bog_rate": float(latest_data['bog_rate'].iloc[0]), "distance": float(latest_data['distance'].iloc[0])}
            )
            new_data.append(latest_data)
        
        self.lng_data = pd.concat([self.lng_data] + new_data, ignore_index=True)
        logger.info(f"Collected data for {len(new_data)} storage facilities")
        return self.lng_data

    def fetch_historical_data(self, limit=100):
        """Retrieve historical data from in-memory DataFrame."""
        logger.info(f"Fetching historical data (limit={limit})")
        if self.lng_data.empty:
            logger.warning("Historical data empty, initializing default")
            self.initialize_default_data()
        return self.lng_data.tail(limit)

    def optimize_route(self, current_speed, current_distance):
        """Optimize ship speed using PuLP."""
        logger.info(f"Optimizing route with speed={current_speed}, distance={current_distance}")
        prob = LpProblem("Route_Optimization", LpMinimize)
        speed_adj = LpVariable("speed_adj", lowBound=-5, upBound=5)
        prob += (current_speed + speed_adj)**2, "Minimize_Fuel_Consumption"
        time_to_dest = current_distance / (current_speed + speed_adj)
        prob += time_to_dest <= current_distance / (self.thresholds['route_fuel_min'] / 100), "Max_Time_Constraint"
        prob.solve()
        if LpStatus[prob.status] == 'Optimal':
            return current_speed + value(speed_adj)
        return current_speed

    @timeout(30)
    def run_bog_agent(self, data):
        """Run BOG Monitoring Agent with timeout."""
        logger.info("Running BOG Agent")
        if data.empty:
            logger.warning("DataFrame is empty, initializing default data")
            data = self.initialize_default_data()
        input_str = ""
        for storage_id in self.storage_facilities:
            latest = data[data['storage_id'] == storage_id].iloc[-1] if not data[data['storage_id'] == storage_id].empty else data.iloc[-1]
            input_str += f"Storage {storage_id}: Temperature: {latest['temperature']}C, Pressure: {latest['pressure']} bar, BOG Rate: {latest['bog_rate']}%/day, Emissions: {latest['emissions']}, Storage: {latest['storage_level']} m3\n"
        try:
            result = self.bog_executor.invoke({
                "input": input_str,
                "tools": self.tools_description,
                "tool_names": self.tool_names
            })
            decision = result.get("output", "No decision")
            for storage_id in self.storage_facilities:
                latest = data[data['storage_id'] == storage_id].iloc[-1] if not data[data['storage_id'] == storage_id].empty else data.iloc[-1]
                if ("anomaly" in decision.lower() or 
                    "high bog" in decision.lower() or 
                    latest['emissions'] > self.thresholds['emissions_max'] or 
                    latest['storage_level'] < self.thresholds['storage_min']):
                    self.shared_context["actions"].append(f"BOG, emission, or storage anomaly detected for {storage_id}")
                    self.agent_errors = 0
                    return decision, f"Route and Cargo Agents notified for {storage_id}"
            self.agent_errors = 0
            return decision, None
        except TimeoutError:
            logger.error("BOG Agent timed out")
            self.agent_errors += 1
            return "BOG Agent timed out", None
        except Exception as e:
            logger.error(f"BOG Agent error: {e}")
            self.agent_errors += 1
            return "BOG Agent failed", None

    @timeout(30)
    def run_route_agent(self, data, bog_decision):
        """Run Route Optimization Agent with timeout."""
        logger.info("Running Route Agent")
        if data.empty:
            logger.warning("DataFrame is empty, initializing default data")
            data = self.initialize_default_data()
        input_str = ""
        for carrier in self.carriers:
            latest = data.iloc[-1]
            input_str += f"Carrier {carrier['id']}: Speed: {latest['speed']} knots, Distance: {latest['distance']} nm, Latitude: {latest['latitude']}, Longitude: {latest['longitude']}, BOG Status: {bog_decision}\n"
        try:
            result = self.route_executor.invoke({
                "input": input_str,
                "tools": self.tools_description,
                "tool_names": self.tool_names
            })
            decision = result.get("output", "No decision")
            optimized_speeds = {}
            for carrier in self.carriers:
                latest = data.iloc[-1]
                if "reroute" in decision.lower() or "adjust speed" in decision.lower():
                    optimized_speed = self.optimize_route(latest['speed'], latest['distance'])
                    optimized_speeds[carrier['id']] = optimized_speed
                    self.shared_context["actions"].append(f"Route adjusted to {optimized_speed} knots for {carrier['id']}")
            self.agent_errors = 0
            return decision, optimized_speeds or None
        except TimeoutError:
            logger.error("Route Agent timed out")
            self.agent_errors += 1
            return "Route Agent timed out", None
        except Exception as e:
            logger.error(f"Route Agent error: {e}")
            self.agent_errors += 1
            return "Route Agent failed", None

    @timeout(30)
    def run_cargo_agent(self, data, bog_decision, route_decision):
        """Run Cargo Scheduling Agent with timeout."""
        logger.info("Running Cargo Agent")
        if data.empty:
            logger.warning("DataFrame is empty, initializing default data")
            data = self.initialize_default_data()
        historical_data = self.fetch_historical_data().to_json()
        storage_forecasts = {}
        for storage_id in self.storage_facilities:
            try:
                logger.info(f"Forecasting for storage_id={storage_id}")
                # Call forecast_storage_level with a single dictionary for LangChain compatibility
                tool_input = {"historical_data": historical_data, "storage_id": storage_id}
                storage_forecasts[storage_id] = forecast_storage_level.run(tool_input)
            except Exception as e:
                logger.error(f"Error forecasting for storage_id={storage_id}: {e}")
                self.agent_errors += 1
                storage_forecasts[storage_id] = pd.DataFrame().to_json()  # Fallback to empty forecast
                if self.agent_errors >= self.max_agent_errors:
                    logger.error("Max agent errors reached in run_cargo_agent")
                    return "Cargo Agent failed due to max errors"
        input_str = f"BOG Status: {bog_decision}, Route: {route_decision}, Storage Forecasts: {json.dumps(storage_forecasts)}, Cargos: {json.dumps(self.cargos)}, Carriers: {json.dumps(self.carriers)}"
        try:
            logger.info(f"Cargo agent input: {input_str}")
            result = self.cargo_executor.invoke({
                "input": input_str,
                "tools": self.tools_description,
                "tool_names": self.tool_names
            })
            decision = result.get("output", "No decision")
            if "schedule cargo" in decision.lower():
                schedule = optimize_cargo_schedule.run(
                    demands=json.dumps(self.cargos),
                    storage_forecasts=json.dumps(storage_forecasts),
                    carriers=json.dumps(self.carriers)
                )
                self.shared_context["forecasts"] = storage_forecasts
                self.shared_context["cargo_schedule"] = schedule
                self.shared_context["actions"].append(f"Cargo scheduled: {schedule}")
            self.agent_errors = 0
            return decision
        except TimeoutError:
            logger.error("Cargo Agent timed out")
            self.agent_errors += 1
            return "Cargo Agent timed out"
        except Exception as e:
            logger.error(f"Cargo Agent error: {e}")
            self.agent_errors += 1
            return "Cargo Agent failed"

    def act(self, bog_decision, route_decision, route_opt, cargo_decision):
        """Execute actions."""
        logger.info(f"Acting on decisions: BOG={bog_decision}, Route={route_decision}, Cargo={cargo_decision}")
        if "anomaly" in bog_decision.lower() or "high bog" in bog_decision.lower():
            alert = f"ALERT: {bog_decision}. Route action: {route_decision}, Speeds: {route_opt}, Cargo: {cargo_decision}"
            self.alerts.append(alert)
            logger.info(alert)
        else:
            logger.info(f"BOG: {bog_decision}, Route: {route_decision}, Cargo: {cargo_decision}")

    def learn(self):
        """Adjust thresholds based on outcomes."""
        logger.info("Learning from outcomes")
        if len(self.alerts) > 0:
            self.thresholds['bog_rate_max'] -= 0.01
            self.thresholds['emissions_max'] -= 1.0
            self.thresholds['storage_min'] += 50
            logger.info(f"Updated thresholds: BOG={self.thresholds['bog_rate_max']}, Emissions={self.thresholds['emissions_max']}, Storage={self.thresholds['storage_min']}")

    def run_loop(self, iterations=3, scenarios=None, max_runtime=300):
        """Main agentic loop with timeout and error limits."""
        start_time = time.time()
        if scenarios is None:
            scenarios = [None]
        logger.info(f"Starting run_loop with {len(scenarios)} scenarios, {iterations} iterations, max_runtime={max_runtime}s")
        for scenario in scenarios:
            if time.time() - start_time > max_runtime:
                logger.error("Max runtime exceeded, stopping loop")
                break
            logger.info(f"Starting scenario: {scenario or 'Normal'}")
            self.lng_data = pd.DataFrame(columns=self.lng_data.columns)
            self.mqtt_message_count = 0
            self.agent_errors = 0
            self.initialize_default_data()
            for i in range(iterations):
                loop_start = time.time()
                if time.time() - start_time > max_runtime:
                    logger.error("Max runtime exceeded, stopping scenario")
                    break
                if self.agent_errors >= self.max_agent_errors:
                    logger.error("Max agent errors reached, stopping scenario")
                    break
                logger.info(f"Iteration {i+1}/{iterations} (Scenario: {scenario or 'Normal'})")
                data = self.collect_data()
                if scenario:
                    data = self.simulate_scenario(scenario, data)
                bog_decision, route_trigger = self.run_bog_agent(data)
                route_decision, route_opt = self.run_route_agent(data, bog_decision)
                cargo_decision = self.run_cargo_agent(data, bog_decision, route_decision)
                self.act(bog_decision, route_decision, route_opt, cargo_decision)
                self.learn()
                logger.info(f"Iteration {i+1} completed in {time.time() - loop_start:.2f}s")
                time.sleep(1)
            historical_data = self.fetch_historical_data()
            self.visualize(historical_data, scenario=scenario)
            self.export_to_csv(scenario=scenario)
        logger.info(f"run_loop completed in {time.time() - start_time:.2f}s")

    def visualize(self, data, scenario=None):
        """Generate interactive Highcharts visualization for multiple storage facilities."""
        logger.info(f"Generating visualization for scenario: {scenario or 'Normal'}")
        if data.empty:
            logger.warning("Visualization data empty, initializing default")
            data = self.initialize_default_data()
        data['time_ms'] = (data['time'].astype(float) * 1000).astype(int)
        
        series = []
        for storage_id in self.storage_facilities:
            storage_data = data[data['storage_id'] == storage_id]
            if not storage_data.empty:
                series.append({
                    'name': f'BOG Rate {storage_id}',
                    'data': list(zip(storage_data['time_ms'], storage_data['bog_rate'])),
                    'yAxis': 0,
                    'color': '#DA291C',
                    'tooltip': {'valueSuffix': '%/day'}
                })
                series.append({
                    'name': f'Temperature {storage_id}',
                    'data': list(zip(storage_data['time_ms'], storage_data['temperature'])),
                    'yAxis': 1,
                    'color': '#00FF00',
                    'tooltip': {'valueSuffix': 'C'}
                })
                series.append({
                    'name': f'Storage Level {storage_id}',
                    'data': list(zip(storage_data['time_ms'], storage_data['storage_level'])),
                    'yAxis': 4,
                    'color': '#800080',
                    'tooltip': {'valueSuffix': 'm3'}
                })
        
        forecast_series = []
        for storage_id, forecast_json in self.shared_context["forecasts"].items():
            forecast_data = pd.read_json(forecast_json) if forecast_json else pd.DataFrame()
            if not forecast_data.empty:
                forecast_data['ds_ms'] = (forecast_data['ds'].astype(int) / 1000000).astype(int)
                forecast_series.append({
                    'name': f'Storage Forecast {storage_id}',
                    'data': list(zip(forecast_data['ds_ms'], forecast_data['yhat'])),
                    'yAxis': 4,
                    'color': '#800080',
                    'dashStyle': 'Dash',
                    'tooltip': {'valueSuffix': 'm3'}
                })
        
        cargo_series = []
        if self.shared_context["cargo_schedule"]:
            schedule = json.loads(self.shared_context["cargo_schedule"])
            for assignment in schedule:
                if assignment['assign']:
                    cargo_series.append({
                        'name': f"{assignment['carrier_id']} -> {assignment['cargo_id']} ({assignment['storage_id']})",
                        'start': data['time_ms'].iloc[-1] + int(assignment['cargo_id'].split('_')[-1]) * 86400000,
                        'end': data['time_ms'].iloc[-1] + (int(assignment['cargo_id'].split('_')[-1]) + 1) * 86400000
                    })
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Shell LNG Transport Monitoring Dashboard</title>
    <script src="https://code.highcharts.com/highcharts.js"></script>
    <script src="https://code.highcharts.com/modules/exporting.js"></script>
    <script src="https://code.highcharts.com/modules/export-data.js"></script>
    <script src="https://code.highcharts.com/gantt.js"></script>
</head>
<body>
    <div id="container" style="width:100%; height:600px;"></div>
    <div id="gantt" style="width:100%; height:400px;"></div>
    <script>
        Highcharts.chart('container', {{
            chart: {{ type: 'line', zoomType: 'x' }},
            title: {{ text: 'Shell LNG Transport Agentic Monitoring ({scenario or "Normal"})' }},
            subtitle: {{ text: 'BOG, Temperature, and Storage Levels Across Facilities' }},
            xAxis: {{ type: 'datetime', title: {{ text: 'Time' }} }},
            yAxis: [{{
                title: {{ text: 'BOG Rate (%/day)' }},
                opposite: false
            }}, {{
                title: {{ text: 'Temperature (C)' }},
                opposite: true
            }}, {{
                title: {{ text: 'Speed (knots)' }},
                opposite: true
            }}, {{
                title: {{ text: 'Emissions (methane/CO2)' }},
                opposite: true,
                plotLines: [{{
                    value: {self.thresholds['emissions_max']},
                    color: '#DA291C',
                    dashStyle: 'Dash',
                    width: 2,
                    label: {{ text: 'Shell Emission Limit' }}
                }}]
            }}, {{
                title: {{ text: 'Storage Level (m3)' }},
                opposite: true,
                plotLines: [{{
                    value: {self.thresholds['storage_min']},
                    color: '#DA291C',
                    dashStyle: 'Dash',
                    width: 2,
                    label: {{ text: 'Min Storage' }}
                }}]
            }}],
            series: {json.dumps(series + forecast_series)},
            tooltip: {{ shared: true }},
            exporting: {{
                enabled: true,
                buttons: {{
                    contextButton: {{
                        menuItems: ["downloadPNG", "downloadPDF", "downloadCSV"]
                    }}
                }}
            }},
            plotOptions: {{
                series: {{
                    point: {{
                        events: {{
                            mouseOver: function() {{
                                if (this.series.name.includes('Storage Level') && this.y < {self.thresholds['storage_min']}) {{
                                    this.series.chart.renderer.label(
                                        'Storage Below Minimum',
                                        this.plotX + this.series.chart.plotLeft,
                                        this.plotY + this.series.chart.plotTop - 20
                                    ).css({{ color: '#DA291C' }}).add();
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }});
        Highcharts.ganttChart('gantt', {{
            title: {{ text: 'LNG Cargo Schedule Across Carriers' }},
            xAxis: {{ type: 'datetime' }},
            yAxis: {{ title: {{ text: 'Carrier -> Cargo' }} }},
            series: [{{
                name: 'Cargo Schedule',
                data: {json.dumps(cargo_series)}
            }}]
        }});
    </script>
</body>
</html>
"""
        filename = f"lng_visualization_{scenario or 'normal'}.html"
        with open(filename, 'w') as f:
            f.write(html_content)
        logger.info(f"Highcharts visualization saved to {filename}")

    def export_to_csv(self, scenario=None):
        """Export data to CSV for Shell reporting."""
        filename = f"lng_data_{scenario or 'normal'}.csv"
        self.lng_data.to_csv(filename, index=False)
        logger.info(f"Data exported to {filename}")

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'mqtt_client') and self.mqtt_client is not None:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            logger.info("MQTT client stopped and disconnected")

# Usage
if __name__ == "__main__":
    agent_system = LNGTransportAgenticSystem()
    agent_system.run_loop(iterations=3, scenarios=['heat_leak'], max_runtime=300)