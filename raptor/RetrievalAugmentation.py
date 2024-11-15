import logging
import pickle

from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import BaseEmbeddingModel
from .QAModels import BaseQAModel, GPT3TurboQAModel, HCX_003_QAModel
from .SummarizationModels import BaseSummarizationModel
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

# Define a dictionary to map supported tree builders to their respective configs
supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalAugmentationConfig:
    def __init__(
        self,
        tree_builder_config=None,
        tree_retriever_config=None,  # Change from default instantiation
        qa_model=None,
        embedding_model=None,
        summarization_model=None,
        tree_builder_type="cluster",
        # New parameters for TreeRetrieverConfig and TreeBuilderConfig
        # TreeRetrieverConfig arguments
        tr_tokenizer=None,
        tr_threshold=0.5,
        tr_top_k=5,
        tr_selection_mode="top_k",
        tr_context_embedding_model="OpenAI",
        tr_embedding_model=None,
        tr_num_layers=None,
        tr_start_layer=None,
        # TreeBuilderConfig arguments
        tb_clustering_algorithm=None,  # 추가
        tb_clustering_params = {},
        tb_tokenizer=None,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_summarization_model=None,
        tb_embedding_models=None,
        tb_cluster_embedding_model="OpenAI",
        tb_metadata_extract_model=None,
    ):
        # Validate tree_builder_type
        if tree_builder_type not in supported_tree_builders:
            raise ValueError(
                f"tree_builder_type must be one of {list(supported_tree_builders.keys())}"
            )

        # Validate qa_model
        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")

        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        elif embedding_model is not None:
            if tb_embedding_models is not None:
                raise ValueError(
                    "Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both."
                )
            tb_embedding_models = {"EMB": embedding_model}
            tr_embedding_model = embedding_model
            tb_cluster_embedding_model = "EMB"
            tr_context_embedding_model = "EMB"

        if summarization_model is not None and not isinstance(
            summarization_model, BaseSummarizationModel
        ):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )

        elif summarization_model is not None:
            if tb_summarization_model is not None:
                raise ValueError(
                    "Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both."
                )
            tb_summarization_model = summarization_model

        # Set TreeBuilderConfig
        tree_builder_class, tree_builder_config_class = supported_tree_builders[
            tree_builder_type
        ]
        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                embedding_models=tb_embedding_models,
                cluster_embedding_model=tb_cluster_embedding_model,
                metadata_extract_model = tb_metadata_extract_model,
                clustering_algorithm=tb_clustering_algorithm,  # 추가
                clustering_params=tb_clustering_params,
            )

        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'"
            )

        # Set TreeRetrieverConfig
        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model,
                embedding_model=tr_embedding_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer,
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError(
                "tree_retriever_config must be an instance of TreeRetrieverConfig"
            )

        # Assign the created configurations to the instance
        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model or GPT3TurboQAModel()
        self.tree_builder_type = tree_builder_type

    def log_config(self):
        config_summary = """
        RetrievalAugmentationConfig:
            {tree_builder_config}
            
            {tree_retriever_config}
            
            QA Model: {qa_model}
            Tree Builder Type: {tree_builder_type}
        """.format(
            tree_builder_config=self.tree_builder_config.log_config(),
            tree_retriever_config=self.tree_retriever_config.log_config(),
            qa_model=self.qa_model,
            tree_builder_type=self.tree_builder_type,
        )
        return config_summary


class RetrievalAugmentation:
    """
    A Retrieval Augmentation class that combines the TreeBuilder and TreeRetriever classes.
    Enables adding documents to the tree, retrieving information, and answering questions.
    """

    def __init__(self, config=None, tree=None):
        """
        Initializes a RetrievalAugmentation instance with the specified configuration.
        Args:
            config (RetrievalAugmentationConfig): The configuration for the RetrievalAugmentation instance.
            tree: The tree instance or the path to a pickled tree file.
        """
        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError(
                "config must be an instance of RetrievalAugmentationConfig"
            )

        # Check if tree is a string (indicating a path to a pickled tree)
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model
        self.clustering_algorithm = config.tree_builder_config.clustering_algorithm

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info(
            f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}"
        )

    def add_documents(self, docs):
        """
        입력 데이터와 토크나이저 타입에 따라 적절한 트리 생성 방법을 선택합니다.
        
        Args:
            docs (Union[str, dict, list]): 
                - 일반 텍스트 문자열 또는
                - FinQA 형식의 JSON 데이터 (dict 또는 list)
        """
        logger = logging.getLogger(__name__)
        
        if self.tree is not None:
            user_input = input(
                "Warning: Overwriting existing tree. Did you mean to call 'add_to_existing' instead? (y/n): "
            )
            if user_input.lower() == "y":
                return

        # FinRAG clustering 사용 여부 확인
        is_using_finrag = self.clustering_algorithm == "FinRAG_Clustering"
        logger.info(f"{self.clustering_algorithm}")

        # 입력 데이터 타입 확인
        is_finqa_data = isinstance(docs, list)
        
        # FinRAG clustering을 사용하면 무조건 FinQA 토크나이저를 사용해야 함
        if is_finqa_data:
            logger.info("FinRAG Clustering 감지: FinQA 토크나이저 사용")
            try:
                self.tree = self.tree_builder.build_from_finqa(input_data=docs)
                logger.info("FinQA 데이터로 트리 생성 완료")
            except Exception as e:
                logger.error(f"FinQA 트리 생성 중 오류 발생: {str(e)}")
                return None
        
        else:
            # 기존 로직 유지
            if not isinstance(docs, str):
                try:
                    docs = str(docs)
                    logger.warning(f"입력 데이터({type(docs)})를 문자열로 변환했습니다.")
                except:
                    logger.error("입력 데이터를 문자열로 변환할 수 없습니다.")
                    return None
                    
            logger.info("일반 텍스트 형식 감지: build_from_text 사용")
            try:
                self.tree = self.tree_builder.build_from_text(input_data=docs)
                logger.info("텍스트 데이터로 트리 생성 완료")
            except Exception as e:
                logger.error(f"트리 생성 중 오류 발생: {str(e)}")
                return None
        
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        logger.info("TreeRetriever 초기화 완료")
        
        return self.tree

    def retrieve(
        self,
        question,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_metadata: bool = False,
        format_output: bool = False,
    ):
        """정보를 검색하고 포맷팅합니다."""
        if self.retriever is None:
            raise ValueError(
                "The TreeRetriever instance has not been initialized. Call 'add_documents' first."
            )
        
        try:
            # TreeRetriever의 retrieve 메서드 호출
            result = self.retriever.retrieve(
                question,
                start_layer=start_layer,
                num_layers=num_layers,
                top_k=top_k,
                max_tokens=max_tokens,
                collapse_tree=collapse_tree,
                return_metadata=True,  # 항상 메타데이터를 가져옵니다
                format_output=format_output,
            )
            
            # format_output이 True인 경우 이미 포맷팅된 문자열이 반환됨
            if format_output:
                return result
            
            # 그렇지 않은 경우 context와 node_info를 적절히 반환
            context, node_info = result
            if return_metadata:
                return context, node_info
            return context
            
        except Exception as e:
            logging.error(f"Error in retrieve: {str(e)}")
            if return_metadata:
                return "", []
            return ""

    def answer_question(
        self,
        question,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_metadata: bool = False,
    ):
        """질문에 대한 답변을 생성합니다."""
        search_question = self.qa_model.generate_search_question(question)
        logging.info(f"질문을 해결하기 위한 검색어: {search_question}")
        
        # 한 번의 retrieve 호출로 필요한 모든 정보를 가져옵니다
        context, node_info = self.retrieve(
            search_question,
            start_layer=start_layer,
            num_layers=num_layers,
            top_k=top_k,
            max_tokens=max_tokens,
            collapse_tree=collapse_tree,
            return_metadata=True  # 항상 메타데이터를 가져옵니다
        )
        
        answer = self.qa_model.answer_question(context, question)
        
        if return_metadata:
            return answer, node_info
        return answer

    def save(self, path):
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {path}")
