from jschon import create_catalog, JSONSchema, URI, JSON

catalog = create_catalog('2020-12')

# this schema is aimed at homogenizing the structure that ml projects are reported


# this is deliberately redundant possibly inconsistent and more semantic than a normalized db schema
# the goal is to version, make reproducable and comparable the multiple diverse ml experiments

# implementers and researchers are expected to verify soundness and add relations between e.g. an artifact and an implementation
# implementers and researchers are also expected to fill in the blank properties

# the resulting json should be released as parseable metadata alongside the model file
MLProjectSchema = JSONSchema({
    "type": "object",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/flowpoint/secora/model.schema.json",
    "title": "MLProjectSchema",
    "description": "Json document that describes a ml project",
    "properties": {
        "MLProjectSchemaVersion": {
            "type": "string",
            },
        "projectName": {
            "description": "a short and memorable name for the project",
            "type": "string"
            },
        "projectId": {
            "description": "an uri that uniquely identifies the project",
            "type": "string"
            },
        "projectHome": {
            "description": "the projects homepage",
            "type": "string"
            },
        "projectVersion": {
            "description": "a unique identifier for a single version of the project",
            "type": "string"
            },
        "projectParameters": {
            "description": "general configurables and parameters that are common amongst ml projects for comparison and hpaam tuning",
            "type": "object",
            "properties": {
                "ModelParameters": {
                    "description": "",
                    "type": "object",
                    "properties": {
                        }
                    },
                "DataParameters": {
                    "description": "",
                    "type": "object",
                    "properties": {
                        }
                    },
                "OptimizerParameters": {
                    "description": "configuration of the optimizer",
                    "type": "object",
                    "properties": {
                        }
                    },
                "MetricsParameters": {
                    "description": "parameters about metrics collection for reproducable metrics",
                    "type": "object",
                    "properties": {
                        }
                    },
                "HyperparamSearch": {
                    "description": "configuration of the hyperparam/nas algorithm",
                    "type": "object",
                    "properties": {
                        }
                    },
                },
            },
        "ProjectArtifacts": {
            "description": "the resulting machine readable files",
            "type": "object",
            "properties": {
                "ModelFiles": {
                    "description": "Files that contain the ml Model, if possible in Onnx format",
                    "type": "object",
                    "properties": {
                        }
                    },
                "LoggedTrainingMetrics": {
                    "description": "metrics of the training process",
                    "type": "object",
                    "properties": {
                        }
                    },
                }
            },
        "ProjectData": {
            "description": "the input data",
            "type": "object",
            "properties": {
                "Datasets": {
                    "description": "Files that contain the ml Model, if possible in Onnx format",
                    "type": "object",
                    "properties": {
                        }
                    },
                }
            },
        "ProjectImplementation": {
            "description": "the actual code",
            "type": "object",
            "properties": {
                "repository": {
                    "description": "the origin version control repository and commit",
                    "type": "string",
                    "properties": {
                        }
                    },
                "environment": {
                    "description": "specific configurables that only matter for running the code",
                    "type": "string",
                    "properties": {
                        }
                    }
                }
            }
        },
    "required": ["MLProjectSchemaVersion", "projectName", "projectId", "projectVersion", "projectParameters", "ProjectArtifacts" ]
    })

MLProjectSchema.validate()

doc = JSON({})
result = MLProjectSchema.evaluate(doc)
valid = result.valid

#so 5 project subdomains:

#configurables/ shared parameters amongst multiple ml projects (like learning rate)
# e.g. is lr a strong concept that doesnt belong to a single optimizer, tbd.

#input aka dataset

#output aka artifacts aka model checkpoints and metrics

#implementation aka fine grained code

# dont save in release/json
# implementation details

# environmentconfig
# debug-loggingconfig
# loggingconfig


# save and release these alongside the model!
# 

# modelconfig
# hparamconfig
# dataconfig
# trainingconfig
# metricconfig

'''
logdir: "/home/slow4/secora_output"
checkpoint_dir: "/home/slow4/secora_output"

# the name has to follow the naming scheme
# see secora/train.py
name: 'run_secora_t0_utc00'
seed: 42
max_checkpoints: 10

num_gpus: 1 #'auto'
amp: default
cuda_graphs: False

preprocess_cores: 10
preprocess_mode: concat
max_input_tokens: 256
languages:
  - all

epochs: 2
shards: 16
warmup_batches: 10000

finetune_mode: all

#model_name: 'microsoft/codebert-base'
model_name: 'roberta-base'
optimizer: adam
lr_schedule: constant

learning_rate: 1e-5
batch_size: 8
grad_accum: 64 # counted in batches
temp: 0.05
dropout: 0.1
grad_clip: 0.0

embedding_size: 768
top_k:  1000
'''
