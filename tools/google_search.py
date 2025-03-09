from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents.tools import Tool


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


class GoogleSearching:
    def __init__(self):
        print(f"Initializing GoogleSearching")
        self.search = GoogleSearchAPIWrapper()
        self.tool = Tool(
            name="Google Search Snippets",
            description="Search Google for recent results.",
            func=self.top1_results,
        )

    def top1_results(self, query):
        return self.search.results(query, 1)

    @prompts(
        name="GoogleSearching",
        description="useful when you need find various information on the Internet. People can use Google Search to solve various problems and needs, such as finding specific information and knowledge, searching for product or service information, looking up news events, finding entertainment content, locating the position and navigation information of specific locations, and more. "
        "The input to this tool should be a string, representing the question",
    )
    def inference(self, inputs):
        # image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        question = inputs
        answer = self.tool.run(question)
        # print(f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
        #       f"Output Answer: {answer}")
        print(
            f"\nProcessed GoogleSearching, Input Question: {question}, "
            f"Output Answer: {answer}"
        )
        return str(answer).replace("{", "").replace("}", "")


if __name__ == '__main__':
    # Define your Google API key
    # google_api_key = "AIzaSyC555h4-H_NzR5WiCeKoAm73v_1ee6c0oc"

    # Initialize the GoogleSearching class with the API key
    google_searching = GoogleSearching()

    # Define a query
    query = "What is the capital of France?"

    # Use the inference method to get the top search result
    result = google_searching.inference(query)

    # Print the result
    print(f"Search result for '{query}': {result}")
