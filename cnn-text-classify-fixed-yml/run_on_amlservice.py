import os

from tools.amlservice_scaffold.amlservice_pipeline import Module, PipelineStep, run_pipeline
from azureml.core import RunConfiguration, Workspace
from azureml.core.environment import DEFAULT_GPU_IMAGE

MODULE_SPECS_FOLDER = 'module_specs'

class TextCNNConstants:
    WORKSPACE_NAME = 'cus-test-cs'
    SUBSCRIPTION_ID = 'e9b2ec51-5c94-4fa8-809a-dc1e695e4896'
    RESOURCE_GROUP = 'cus-test-cs'
    COMPUTE_NAME = 'gpu0'


def spec_file_path(spec_file_name):
    return os.path.join(MODULE_SPECS_FOLDER, spec_file_name)


def get_workspace(name, subscription_id, resource_group):
    return Workspace.get(
        name=name,
        subscription_id=subscription_id,
        resource_group=resource_group
    )


def get_run_config(comp, compute_name, use_gpu=False):
    if comp.image:
        run_config = RunConfiguration()
        run_config.environment.docker.base_image = comp.image
    else:
        run_config = RunConfiguration(conda_dependencies=comp.conda_dependencies)
    run_config.target = compute_name
    run_config.environment.docker.enabled = True
    if use_gpu:
        run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
        run_config.environment.docker.gpu_support = True

    return run_config



def create_pipeline_steps(compute_name):
    # Load module spec from module_specs file
    import_data = Module(
        spec_file_path=spec_file_path('import_data.yaml'),
        source_directory='csmodel'
    )

    import_data_test = Module(
        spec_file_path=spec_file_path('import_data2.yaml'),
        source_directory='csmodel'
    )

    train = Module(
        spec_file_path=spec_file_path('train.yaml'),
        source_directory='csmodel'
    )

    preprocess_module = Module(
        spec_file_path=spec_file_path('preprocess_data.yaml'),
        source_directory='csmodel'
    )
    score = Module(
        spec_file_path=spec_file_path('score.yaml'),
        source_directory='csmodel'
    )

    run_config_import_train_data = get_run_config(import_data, compute_name)
    run_config_import_test_data = get_run_config(import_data_test, compute_name)
    run_config_preprocess_test_data = get_run_config(preprocess_module, compute_name)
    run_config_train = get_run_config(train, compute_name, use_gpu=True)
    run_config_score = get_run_config(score, compute_name, use_gpu=True)


    import_data.params['command json'].assign('import_data.json')
    import_data_test.params['command json'].assign('import_data2.json')


    train.inputs['Train data file'].connect(import_data.outputs['9e5eade8_fa58_4a3b_9c51_d9b3f704b756'])
    train.inputs['Validation data file'].connect(import_data_test.outputs['9e5eade8_fa58_4a3b_9c51_d9b3f704b123'])



    # Assign parameters
    train.params['Label column name'].assign('label')
    train.params['Text column name'].assign('text')
    train.params['Word embedding dim'].assign(300)
    # train.params['Kernel size'].assign('3,4,5')
    train.params['Kernel num'].assign(256)
    train.params['Dropout'].assign(0.5)
    train.params['Batch size'].assign(128)
    train.params['Epoch num'].assign(1)
    train.params['L2 regularization weight'].assign(0.)
    train.params['Test interval'].assign(100)

    preprocess_module.inputs['Input vocab'].connect(train.outputs['Vocab'])
    preprocess_module.inputs['Input text'].connect(import_data_test.outputs['9e5eade8_fa58_4a3b_9c51_d9b3f704b123'])

    score.inputs['Trained model'].connect(train.outputs['Trained model'])
    score.inputs['Predict data'].connect(preprocess_module.outputs["Processed data"])

    pipeline_step_list = [
        PipelineStep(import_data, run_config=run_config_import_train_data),
        PipelineStep(import_data_test, run_config=run_config_import_test_data),
        PipelineStep(train, run_config=run_config_train),
        PipelineStep(preprocess_module, run_config=run_config_preprocess_test_data),
        PipelineStep(score, run_config=run_config_score)
    ]
    return pipeline_step_list

if __name__ == '__main__':
    workspace = get_workspace(
        name=TextCNNConstants.WORKSPACE_NAME,
        subscription_id=TextCNNConstants.SUBSCRIPTION_ID,
        resource_group=TextCNNConstants.RESOURCE_GROUP
    )
    compute_name = TextCNNConstants.COMPUTE_NAME
    pipeline_steps = create_pipeline_steps(compute_name)
    run_pipeline(steps=pipeline_steps, experiment_name='cnn-text-classify_new', workspace=workspace)
