import { layers, sequential, Sequential, tensor2d } from "@tensorflow/tfjs-node";
import { TDataset } from "./types.js";
import { inputEncoder, outputEncoder } from "./utils/encoder.js";
import dataset from "./data/training_data.json" with { type: "json" };

const createModel = async () => {
    const model = sequential();

    model.add(layers.dense({
        inputShape: [512],
        activation: 'sigmoid',
        units: 2,
    }));

    model.add(layers.dense({
        inputShape: [2],
        activation: 'sigmoid',
        units: 2,
    }));

    model.add(layers.dense({
        inputShape: [2],
        activation: 'sigmoid',
        units: 2,
    }));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'adam',
    });

    return model;
}

const trainModel = async (model: Sequential, data: TDataset[], epochs: number) => {
    const inputDataset = data.map(ele => ele.input);
    const input = await inputEncoder(inputDataset);
    const output = await outputEncoder(data);
    await model.fit(tensor2d(input), output, { epochs, shuffle: true, batchSize: 64 });
}

const run = async () => {
    const model = await createModel();
    await trainModel(model, dataset, 1000)
    await model.save('file://model');
}

run();