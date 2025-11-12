from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("ryk012/ctb_dataset", tag="_version_", repo_type="dataset")
