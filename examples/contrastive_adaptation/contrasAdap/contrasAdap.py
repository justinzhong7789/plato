"""
The implementation for our contrastive adaptation method.


"""

import contrasAdap_net
import contrasAdap_server
import contrasAdap_client
import contrasAdap_trainer
import contrasAdap_algorithm


def main():
    """ A Plato federated learning training session using the SimCLR algorithm.
        This implementation of simclr utilizes the general setting, i.e.,
        removing the final fully-connected layers of model defined by
        the 'model_name' in config file.
    """
    algorithm = contrasAdap_algorithm.Algorithm
    trainer = contrasAdap_trainer.Trainer
    contras_adap_model = contrasAdap_net.ContrasAdap
    client = contrasAdap_client.Client(model=contras_adap_model,
                                       trainer=trainer,
                                       algorithm=algorithm)
    server = contrasAdap_server.Server(model=contras_adap_model,
                                       trainer=trainer,
                                       algorithm=algorithm)

    server.run(client)


if __name__ == "__main__":
    main()
