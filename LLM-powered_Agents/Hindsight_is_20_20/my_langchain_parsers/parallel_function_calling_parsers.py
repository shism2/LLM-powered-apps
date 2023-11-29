class MyAIMessageToAgentActionParserForParallelFunctionCalling(JsonOutputToolsParser):
    tools: List[Type[BaseModel]]    
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:

        generation = result[0] if isinstance(result, list) else result
        if generation.text:
            log = generation.text
            return_values = {'output':log}
            return AgentFinish(return_values=return_values, log=log, type='AgentFinish')
        else:
            generation = generation.message
            tool_calls = copy.deepcopy(generation.additional_kwargs["tool_calls"])
            results = []
            for tool_call in tool_calls:
                if "function" not in tool_call:
                    pass
                function_args = tool_call["function"]["arguments"]
                results.append(
                    {
                        "type": tool_call["function"]["name"],
                        "args": json.loads(function_args),
                    }
                )
            tool = [res["type"] for res in results]
            name_dict = {tool.__name__: tool for tool in self.tools}
            
            args = [name_dict[res["type"]](**res["args"]) for res in results]
            log = '\nInvoking: ['+', '.join([x+'('+str(y)+')' for x, y in zip(tool, args)])+']'
            return AgentActionMessageLog(log=log, tool=', '.join(tool), tool_input={'tool_input': [(res["type"],res["args"]) for res in results] }, message_log=[generation])