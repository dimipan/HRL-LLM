import json
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.document_loaders import JSONLoader
from pprint import pprint
from utils_functions import get_file_type


class DisasterResponseAssistant:
    def __init__(self, data_path, data_type, model_name="mistral", embedding_model='nomic-embed-text', collection_name="rag-chroma"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.data_path = data_path
        self.data_type = data_type
        
        self.llm = None
        self.loader = None
        self.vectorstore = None
        self.retriever = None
        self.after_rag_chain = None  # To hold the entire RAG chain
        
        self._load_model()            # Initializes an instance of the ChatOllama model    
        self._load_documents()        # Loads and splits the document into chunks
        self._create_vectorstore()    # Creates a vector store using Chroma from the document splits
        self._create_retriever()      # Creates a retriever from the vector store
        self._initialize_rag_chain()  # Initializes the RAG chain with the retriever
       
    
    def _load_model(self):
        self.llm = ChatOllama(model=self.model_name)

    def _load_model(self):
        self.llm = ChatOllama(model=self.model_name)

    def _load_documents(self): ## for json documents
        print(f"document {self.data_type} will be infused")
        if self.data_type == 'pdf':
            self.loader = PyPDFLoader(self.data_path)
            self.data = self.loader.load_and_split()
        elif self.data_type == 'json':
            self.loader = JSONLoader(
                file_path=self.data_path,
                jq_schema='.',
                text_content=False)
            self.data = self.loader.load()
            #pprint(self.data)
        else:
            raise ValueError("Unsupported document type. Please choose either 'pdf' or 'json'.")

    def _create_vectorstore(self): ## for json documents
        self.vectorstore = Chroma.from_documents(
            documents=self.data,
            collection_name=self.collection_name,
            embedding=embeddings.ollama.OllamaEmbeddings(model=self.embedding_model),
        )

    def _create_retriever(self):
        self.retriever = self.vectorstore.as_retriever()
        
    def _initialize_rag_chain(self):
        prompt_template = """You are an assistant, who carefully listens to verbal inputs: {verbal_input} and specialized in analyzing disaster-related inputs. Your task is 
        to identify physical locations mentioned in the text and classify them as either points of interest (POI) or as hazards/dangers (HAZARD) for rescue operations. Use the
        information provided in the documents: {context}, such as KEYWORDS, descriptions and context when locations are mentioned, to make your classification.
        Output the classification in the form of a JSON array dictionary with keys 'location', 'coordinates', and 'category'. Here are some rules you always follow:
        - Focus strictly on physical locations. Avoid including entities that do not represent physical, geographical places (such as individuals, conditions, or 
          abstract concepts).
        - Generate human-readable output in the specified dictionary format.
        - Understand and identify the actual physical location when synonyms or descriptive phrases are used in the {verbal_input}.
        - Generate only the requested output, strictly following the dictionary structure.
        - Within the dictionary, the value of the `category` key must be either 'POI' or 'HAZARD'. 
        - Never generate offensive or foul language.
        - Never give explanations over your output.
        Input: {verbal_input}
        """
        system_template = ChatPromptTemplate.from_template(prompt_template)
        output_parser = StrOutputParser()
        self.after_rag_chain = (
            {"context": self.retriever, "verbal_input": RunnablePassthrough()}
            | system_template
            | self.llm  
            | output_parser
        )
    
    def generate_response(self, verbal_input):
        response = self.after_rag_chain.invoke(verbal_input)
        return response

    def refine_response(self, output):
        cleaned_output_str = output.strip().replace('\n', '').replace('(', '[').replace(')', ']')
        output_dict = json.loads(cleaned_output_str)
        hazard_coordinates, poi_coordinates = [], []
        for item in output_dict:
            coord = tuple(item['coordinates'])
            if item['category'] == 'HAZARD':
                hazard_coordinates.append(coord)
            else:
                poi_coordinates.append(coord)
                    
        print("Hazardous Coordinates:", hazard_coordinates)
        print("Point of Interest Coordinates:", poi_coordinates)
        return hazard_coordinates, poi_coordinates
    
    
### ----- E X A M P L E -----    
#document_path = "/home/research100/Documents/sample/enhanced_RL/enhanced_RL/data/sar_data.pdf"
document_path = '/home/research100/Documents/sample/enhanced_RL/enhanced_RL/data/sar_data.json'     
document_type = get_file_type(document_path)
assistant = DisasterResponseAssistant(document_path, document_type)

input = "Hey, there's a victim at the medical centre. I also saw fire in the school and at the money building. There's a shelter through the train station."
#print(f"The INPUT is: {input}")
res = assistant.generate_response(input)
print(res)
haz, poi = assistant.refine_response(res)
print(haz)
print(poi)
