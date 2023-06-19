if __name__ == '__main__':
    from evaluate_DT import evaluate

    evaluate(DT_model="TobiTob/decision_transformer_fr_24", buildings_to_use="validation",
             TR=-9000, evaluation_interval=24, simulation_start_end=[0, 8759], file_to_save='_results1.txt')

    evaluate(DT_model="TobiTob/decision_transformer_fr_24", buildings_to_use="test",
             TR=-9000, evaluation_interval=24, simulation_start_end=[0, 8759], file_to_save='_results2.txt')
