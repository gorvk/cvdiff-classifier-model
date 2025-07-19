import * as tf from "@tensorflow/tfjs-node";
import * as use from "@tensorflow-models/universal-sentence-encoder";
import tokenizer from "./tokenizer.js";

const encodeData = async (data) => {
  try {
    const model = await use.load();
    return await model.embed(data);
  } catch (err) {
    console.error("Fit Error:", err);
    return null;
  }
};

const predict = async (testingData) => {
  const model = await tf.loadLayersModel("file://model/model.json");
  const input = await encodeData(testingData);
  const data = await model.predict(input).argMax(1).data();
  data.forEach((x, i) => {
    console.log(testingData[i], x === 0 ? "t_skill" : "o_skill");
  });

  console.log(
    "*********************************************************************************"
  );

  data.forEach((d, i) => {
    if (d === 0) {
      console.log(testingData[i]);
    }
  });

  console.log(
    "*********************************************************************************"
  );

  data.forEach((d, i) => {
    if (d === 1) {
      console.log(testingData[i]);
    }
  });
};

const JD =
  "Develop responsive, user-friendly UI components using HTML, CSS, JavaScript, Typescript and frameworks like Angular, React, or Vue.js. Collaborate with UX/UI designers to create intuitive designs and optimize performance. Ensure consistent design and smooth user experience across devices. Experience with version control systems such as Azure DevOps, Git etc. Participate in code reviews and provide constructive feedback to team members. Experience in developing across multiple environments like Mobile, Tablets etc and Operating systems like Android, IOS etc would be highly desired. Experience in working with SAFe/Agile practices. Participate in user testing and apply feedback to improve interfaces. Write clean, maintainable code and follow best practices Requirements: Proven experience as a UI Frontend Developer with a strong portfolio. Proficiency in HTML5, CSS, JavaScript, and frontend frameworks Angular, React, Vue. Experience with responsive design and cross-browser compatibility. Knowledge of UX/UI principles, accessibility standards, and performance optimization. Familiarity with Git, Azure DevOps. Experience with design tools (Figma, Sketch), TypeScript, and cloud platforms. Familiarity with Agile development and CI/CD processes.";

const X = tokenizer(JD);
predict(X);
