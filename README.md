# Chain of Agents Map Reduce

This project implements a Chain of Agents (CoA) framework with a Map-Reduce architecture for question answering. It leverages language models to process context chunks in parallel and generate final answers.

## Overview

The project consists of the following main components:

*   **CoAMapReduce**: Orchestrates the Map-Reduce process, distributing tasks to worker agents and collecting results.
*   **CoABaseline**: Implements a baseline sequential chain of agents.

## Setup

1.  **Set the API Key**: Set the `GOOGLE_API_KEY` environment variable with your Gemini API key.

    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```

## Usage

To run the map-reduce evaluation:

```bash
python src/core/coa_map_reduce.py --num_samples 10