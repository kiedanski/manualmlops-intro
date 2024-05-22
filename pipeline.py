from kfp import dsl, compiler, local


@dsl.container_component
def preprocessing_op():
    return dsl.ContainerSpec(
        image='northamerica-northeast1-docker.pkg.dev/mlops-devrel/train/train:latest',
        # image="train_docker",
        command=['python'],
        args=['preprocess.py']
    )

@dsl.container_component
def training_op():
    return dsl.ContainerSpec(
        image='northamerica-northeast1-docker.pkg.dev/mlops-devrel/train/train:latest',
        # image="train_docker",
        command=['python'],
        args=['train.py']
    )

@dsl.pipeline
def ml_pipeline():
    preprocessing = preprocessing_op()
    training = training_op().after(preprocessing)

if __name__ == '__main__':
    local.init(runner=local.DockerRunner())
    ml_pipeline()
    # compiler.Compiler().compile(ml_pipeline, 'ml_pipeline.yaml')

