import { loadLayersModel, tensor2d } from "@tensorflow/tfjs-node";
import tokenizer from "./utils/tokenizer.js";
import { inputEncoder } from "./utils/encoder.js";

const predict = async (testingData: string[]) => {
  const model = await loadLayersModel("file://model/model.json");
  const input = await inputEncoder(testingData);
  if (!input) {
    return;
  }

  const prediction = model.predict(tensor2d(input));

  if (!Array.isArray(prediction)) {
    const data = await prediction.argMax(1).data();
    data.forEach((x: number, i: number) => {
      console.log(testingData[i], x === 0 ? "t_skill" : "o_skill");
    });

    console.log(
      "*********************************************************************************"
    );

    data.forEach((x: number, i: number) => {
      if (x === 0) {
        console.log(testingData[i]);
      }
    });

    console.log(
      "*********************************************************************************"
    );

    data.forEach((x: number, i: number) => {
      if (x === 1) {
        console.log(testingData[i]);
      }
    });
  }

};

const JD =
  "Develop responsive, user-friendly UI components using HTML, CSS, JavaScript, Typescript and frameworks like Angular, React, or Vue.js. Collaborate with UX/UI designers to create intuitive designs and optimize performance. Ensure consistent design and smooth user experience across devices. Experience with version control systems such as Azure DevOps, Git etc. Participate in code reviews and provide constructive feedback to team members. Experience in developing across multiple environments like Mobile, Tablets etc and Operating systems like Android, IOS etc would be highly desired. Experience in working with SAFe/Agile practices. Participate in user testing and apply feedback to improve interfaces. Write clean, maintainable code and follow best practices Requirements: Proven experience as a UI Frontend Developer with a strong portfolio. Proficiency in HTML5, CSS, JavaScript, and frontend frameworks Angular, React, Vue. Experience with responsive design and cross-browser compatibility. Knowledge of UX/UI principles, accessibility standards, and performance optimization. Familiarity with Git, Azure DevOps. Experience with design tools (Figma, Sketch), TypeScript, and cloud platforms. Familiarity with Agile development and CI/CD processes.";

const X = tokenizer(JD);
predict(X);
