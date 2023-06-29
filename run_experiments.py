if __name__ == '__main__':
    from evaluate_DT import evaluate

    evaluate(DT_model="TobiTob/decision_transformer_merged1", buildings_to_use="validation",
             TR=-9000, evaluation_interval=168, simulation_start_end=[0, 8759], file_to_save='_results100_9000.txt')
    evaluate(DT_model="TobiTob/decision_transformer_merged1", buildings_to_use="validation",
             TR=-6000, evaluation_interval=168, simulation_start_end=[0, 8759], file_to_save='_results100_6000.txt')
    evaluate(DT_model="TobiTob/decision_transformer_merged1", buildings_to_use="validation",
             TR=-12000, evaluation_interval=168, simulation_start_end=[0, 8759], file_to_save='_results100_12000.txt')

