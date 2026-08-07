from credit_risk.components.data_ingestion import DataIngestion
from credit_risk.components.data_transformation import DataTransformation
from credit_risk.components.model_evaluation import ModelEvaluation
from credit_risk.components.model_trainer import ModelTrainer


def run_training() -> None:
    """Run the legacy end-to-end training workflow explicitly."""
    data_ingestion = DataIngestion()
    train_data_path, test_data_path, train_target_path, test_target_path = (
        data_ingestion.initiate_data_ingestion()
    )

    data_transformation = DataTransformation()
    train_arr, test_arr, train_target, test_target = (
        data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path,
            train_target_path,
            test_target_path,
        )
    )

    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr, train_target)

    model_evaluation = ModelEvaluation()
    model_evaluation.initiate_model_evaluation(
        train_arr,
        test_arr,
        train_target,
        test_target,
    )


if __name__ == "__main__":
    run_training()
