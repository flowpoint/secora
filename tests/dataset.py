from datasets import Dataset
import datasets

VERSION = datasets.Version("0.0.1", "CodeSearchNet corpus synthetic test")
LANGS = ['python', 'java', 'go', 'javascript', 'ruby', 'php']

class CodeSearchNet(datasets.GeneratorBasedBuilder):
    VERSION = VERSION
    LANGS = LANGS
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=lang,
            version=VERSION,
            description="All available languages: Java, Go, Javascript, Python, PHP, Ruby",
        ) for lang in LANGS + ['all']]

    SPLITS = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={}
            ),
        ]

    def _split_generators(self, dl_manager):
        return self.SPLITS

    def _generate_examples(self):
        countr = 0
        for l in LANGS:
            for k in range(1000):
                countr += 1
                sample = countr, {
                       "repository_name": 'https://github.com/test/test',
                       "func_path_in_repository": 'test',
                       "func_name": 'foo',
                       "whole_func_string": "def foo(x):''' a foo function ''' return x",
                       "language": l,
                       "func_code_string": "def foo(x): return x",
                       "func_code_tokens": ["def", "foo", "(", "x", ")", ":", "return", "x"],
                       "func_documentation_string": "a foo function",
                       "func_documentation_tokens": ["a", "foo", "function"],
                       "split_name": 'train',
                       "func_code_url": 'https://github.com/test/test#L1-L2'
                   }
                yield sample

    def _info(self):
        return datasets.DatasetInfo(
            description="syntetic codesearchnet for testing",
            features=datasets.Features(
                {
                    "repository_name": datasets.Value("string"),
                    "func_path_in_repository": datasets.Value("string"),
                    "func_name": datasets.Value("string"),
                    "whole_func_string": datasets.Value("string"),
                    "language": datasets.Value("string"),
                    "func_code_string": datasets.Value("string"),
                    "func_code_tokens": datasets.Sequence(datasets.Value("string")),
                    "func_documentation_string": datasets.Value("string"),
                    "func_documentation_tokens": datasets.Sequence(datasets.Value("string")),
                    "split_name": datasets.Value("string"),
                    "func_code_url": datasets.Value("string"),
                    # TODO - add licensing info in the examples
                }
            ),
            # No default supervised keys
            supervised_keys=None,
        )
