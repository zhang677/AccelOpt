# Test the signoz logging function.
import json
import logfire
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunConfig, ModelSettings
from accelopt.utils import retry_runner_safer
import clickhouse_connect
import asyncio
import time

def extract_thought(text):
    start_tag = "<thought>"
    end_tag = "</thought>"
    
    # Find the indices of the tags
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)
    
    # Check if both tags exist and are in the correct order
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        # Move index to the end of the start_tag to get the content inside
        return text[start_idx + len(start_tag):end_idx].strip()
    
    return None

def fetch_experiment_history(service_name, model_name):
    client = clickhouse_connect.get_client(host='localhost', port=8123)

    query = f"""
    SELECT 
        T.trace_id,
        T.timestamp as start_time,
        T.attributes_number['gen_ai.usage.input_tokens'] as tokens_in,
        T.attributes_number['gen_ai.usage.output_tokens'] as tokens_out,
        T.attributes_string['request_data'] as request_data,
        T.attributes_string['response_data'] as response_data
    FROM signoz_traces.distributed_signoz_index_v3 AS T
    WHERE T.serviceName = '{service_name}' 
        AND T.name LIKE 'Chat Completion with%' 
        AND T.attributes_string['gen_ai.request.model'] = '{model_name}'
        AND T.statusCodeString != 'Error'
    ORDER BY start_time DESC
    LIMIT 20
    """
    
    result = client.query(query)
    
    history = []
    for row in result.result_rows:
        tid, ts, t_in, t_out, request_data, response_data = row
        input_messages = json.loads(request_data)['messages']
        output_message = json.loads(response_data)['message']
        inputs = []
        for msg in input_messages:
            inputs.append({msg['role']: msg['content']})
        output = output_message['content']
        if 'reasoning_content' in output_message:
            thinking = output_message['reasoning_content']
        else:
            print(output_message.keys())
            # Extract thinking if available
            thinking = extract_thought(output)

        

        history.append({
            "time": ts,
            "trace_id": tid,
            "usage": f"{int(t_in or 0)}/{int(t_out or 0)}",
            "input": "\n\n".join([f"[{list(msg.keys())[0]}]: {list(msg.values())[0]}" for msg in inputs]),
            "output": output,
            "thinking": thinking # Add to dictionary
        })
            
    return history

def delete_test_traces():
    service_name='test_signoz'
    client = clickhouse_connect.get_client(host='localhost', port=8123)
    
    # 1. Delete from the Index table
    # The physical column name is 'resource_string_service$$name'
    try:
        print(f"Deleting index entries for {service_name}...")
        client.command(f"""
            ALTER TABLE signoz_traces.signoz_index_v3 
            DELETE WHERE `resource_string_service$$name` = '{service_name}'
        """)
    except Exception as e:
        print(f"Error deleting from index: {e}")

    # 2. Delete from the Spans table
    # This table doesn't have the service column at all, so we match on trace_id.
    # Note: Index table uses 'trace_id', Spans table uses 'traceID'
    try:
        print(f"Deleting span data for {service_name}...")
        client.command(f"""
            ALTER TABLE signoz_traces.signoz_spans 
            DELETE WHERE traceID IN (
                SELECT trace_id FROM signoz_traces.signoz_index_v3 
                WHERE `resource_string_service$$name` = '{service_name}'
            )
        """)
    except Exception as e:
        print(f"Error deleting from spans: {e}")

    print("Cleanup mutations submitted.")

if __name__ == "__main__":
    model_config_path = "../experiments/full_complete_local/configs/planner_config.json"
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    run_query = True
    delete_existing_traces = True
    if delete_existing_traces:
        delete_test_traces()
    if run_query:
        BASE_URL = model_config['url']
        API_KEY = model_config['api_key']
        set_tracing_disabled(disabled=True)
        logfire.configure()
        logfire.instrument_openai()

        client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=10000000)
        model = OpenAIChatCompletionsModel(
            model=model_config["model"],# "openai/gpt-oss-120b",
            openai_client=client
        )

        agent = Agent(
            name="Planner",
            instructions="You are a helpful assistant",
            model=model
        )

        user_prompt = "Imagine you are a human. Is it possible to take a 6m stick and pass it through a tunnel that is 4m high and 3m wide?"

        if 'gemini' in model_config["model"].lower():
            run_config = RunConfig(
                model_settings=ModelSettings(
                    extra_body={
                        'extra_body':{
                            'google': {
                                'thinking_config': {
                                    'include_thoughts': True
                                }
                            }
                        }
                    }
                )
            ) # The response will start with <thought>...</thought>
        else:
            run_config = None

        result = asyncio.run(retry_runner_safer(agent, user_prompt, run_config=run_config))
        print("Waiting for SigNoz to index the trace...")
        time.sleep(10)  # Wait for indexing

    data = fetch_experiment_history("test_signoz", model_config["model"])
    
    for id, item in enumerate(data):
        print(f"[{item['time']}] Trace: {item['trace_id']} id={id}")
        
        # Clean up for summary
        think_preview = str(item['thinking']).replace('\n', ' ')
        
        if item['thinking']:
            print(f"THINKING: {think_preview[:100]}...")
        
        # Clean up newlines for the summary view
        inp_preview = str(item['input']).replace('\n', ' ')
        out_preview = str(item['output']).replace('\n', ' ')
        
        print(f"INPUT:  {inp_preview[:150]}...")
        print(f"OUTPUT: {out_preview[:150]}...")

        print("-" * 80)
