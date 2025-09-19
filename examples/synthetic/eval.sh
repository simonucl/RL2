MODEL=$1

EVAL_ENGINE="vllm"

SERVER_PORT=8000
SERVER_HOST="0.0.0.0"
SERVER_STARTUP_TIMEOUT=120

# Function to start the inference server
start_server() {
    echo "Starting $EVAL_ENGINE server..."
    
    if [ "$EVAL_ENGINE" == "vllm" ]; then
        vllm serve $MODEL \
            --port $SERVER_PORT --max-model-len 32768 \
            --enable-prefix-caching --dtype bfloat16 > vllm.log 2>&1 &
        SERVER_PID=$!
    else
        echo "Error: Unknown evaluation engine '$EVAL_ENGINE'"
        exit 1
    fi
    
    # Wait for server to start up
    echo "Waiting for server to start up (timeout: ${SERVER_STARTUP_TIMEOUT}s)..."
    elapsed=0
    while ! curl -s "http://localhost:8000/models" > /dev/null; do
        sleep 2
        elapsed=$((elapsed + 2))
        echo "Still waiting for server... (${elapsed}s elapsed)"
        
        # Check timeout
        if [ $elapsed -ge $SERVER_STARTUP_TIMEOUT ]; then
            echo "Error: Server startup timeout after ${SERVER_STARTUP_TIMEOUT} seconds"
            echo "Check the server logs for details:"
            if [ "$EVAL_ENGINE" == "vllm" ]; then
                echo "vLLM log: $(pwd)/vllm.log"
            elif [ "$EVAL_ENGINE" == "sglang" ]; then
                echo "SGLang log: $(pwd)/sglang.log"
            fi
            exit 1
        fi
        
        # Check if server process is still running
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Error: Server process died unexpectedly. Check logs."
            exit 1
        fi
    done
    echo "Server is up and running!"
}

# Function to stop the server
stop_server() {
    echo "Stopping $EVAL_ENGINE server..."
    if [ "$EVAL_ENGINE" == "vllm" ]; then
        pkill -f "vllm serve $MODEL --port $SERVER_PORT" || true
    elif [ "$EVAL_ENGINE" == "sglang" ]; then
        pkill -f "python -m sglang.launch_server --model-path $MODEL --host $SERVER_HOST --dp $DATA_PARALLEL_SIZE --port $SERVER_PORT" || true
    fi
    sleep 2
}

start_server
python evals/gsm.py --model $MODEL --vllm_base_url http://localhost:8000 --repeat 5
stop_server
sleep 10