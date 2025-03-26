import { useCallback, useState } from "react";

export const DEFAULT_TEXT_TEMPERATURE = 0.4;
export const DEFAULT_TEXT_TOPK = 25;
export const DEFAULT_AUDIO_TEMPERATURE = 0.7;
export const DEFAULT_AUDIO_TOPK = 250;
export const DEFAULT_PAD_MULT = 0;
export const DEFAULT_REPETITION_PENALTY_CONTEXT = 64;
export const DEFAULT_REPETITION_PENALTY = 1.15;
export const DEFAULT_IMAGE_RESOLUTION = 448;
export const DEFAULT_IMAGE_URL = undefined;
export const DEFAULT_GATE_DELAY = 16;
export const DEFAULT_GATE_INFLUENCE = 0.0;
export const DEFAULT_DISPLAY_COLOR = true;
export const DEFAULT_CENTER_CROP = false;

export type ModelParamsValues = {
  textTemperature: number;
  textTopk: number;
  audioTemperature: number;
  audioTopk: number;
  padMult: number;
  repetitionPenaltyContext: number,
  repetitionPenalty: number,
  imageResolution: number,
  imageUrl: string | undefined,
  gateDelay: number,
  gateInfluence: number,
  displayColor: boolean,
  centerCrop: boolean,
};

export function importantSettingsHaveChanged(params: ModelParamsValues): boolean {
  return (params.textTemperature != DEFAULT_TEXT_TEMPERATURE) ||
    (params.textTopk != DEFAULT_TEXT_TOPK) ||
    (params.audioTemperature != DEFAULT_AUDIO_TEMPERATURE) ||
    (params.audioTopk != DEFAULT_AUDIO_TOPK) ||
    (params.padMult != DEFAULT_PAD_MULT) ||
    (params.repetitionPenalty != DEFAULT_REPETITION_PENALTY) ||
    (params.repetitionPenaltyContext != DEFAULT_REPETITION_PENALTY_CONTEXT) ||
    (params.imageResolution != DEFAULT_IMAGE_RESOLUTION) ||
    (params.gateDelay != DEFAULT_GATE_DELAY) ||
    (params.gateInfluence != DEFAULT_GATE_INFLUENCE) ||
    (params.centerCrop != DEFAULT_CENTER_CROP)
}

type useModelParamsArgs = Partial<ModelParamsValues>;

export const useModelParams = (params?: useModelParamsArgs) => {

  const [textTemperature, setTextTemperatureBase] = useState(params?.textTemperature || DEFAULT_TEXT_TEMPERATURE);
  const [textTopk, setTextTopkBase] = useState(params?.textTopk || DEFAULT_TEXT_TOPK);
  const [audioTemperature, setAudioTemperatureBase] = useState(params?.audioTemperature || DEFAULT_AUDIO_TEMPERATURE);
  const [audioTopk, setAudioTopkBase] = useState(params?.audioTopk || DEFAULT_AUDIO_TOPK);
  const [padMult, setPadMultBase] = useState(params?.padMult || DEFAULT_PAD_MULT);
  const [repetitionPenalty, setRepetitionPenaltyBase] = useState(params?.repetitionPenalty || DEFAULT_REPETITION_PENALTY);
  const [repetitionPenaltyContext, setRepetitionPenaltyContextBase] = useState(params?.repetitionPenaltyContext || DEFAULT_REPETITION_PENALTY_CONTEXT);
  const [imageResolution, setImageResolutionBase] = useState(params?.imageResolution || DEFAULT_IMAGE_RESOLUTION);
  const [imageUrl, setImageUrlBase] = useState(params?.imageUrl || DEFAULT_IMAGE_URL);
  const [gateDelay, setGateDelayBase] = useState(params?.gateDelay || DEFAULT_GATE_DELAY);
  const [gateInfluence, setGateInfluenceBase] = useState(params?.gateInfluence || DEFAULT_GATE_INFLUENCE);
  const [displayColor, setDisplayColorBase] = useState<boolean>(params?.displayColor == undefined ? DEFAULT_DISPLAY_COLOR : params?.displayColor);
  const [centerCrop, setCenterCropBase] = useState<boolean>(params?.centerCrop == undefined ? DEFAULT_CENTER_CROP : params?.centerCrop);

  const resetParams = useCallback(() => {
    setTextTemperatureBase(DEFAULT_TEXT_TEMPERATURE);
    setTextTopkBase(DEFAULT_TEXT_TOPK);
    setAudioTemperatureBase(DEFAULT_AUDIO_TEMPERATURE);
    setAudioTopkBase(DEFAULT_AUDIO_TOPK);
    setPadMultBase(DEFAULT_PAD_MULT);
    setRepetitionPenaltyBase(DEFAULT_REPETITION_PENALTY);
    setRepetitionPenaltyContextBase(DEFAULT_REPETITION_PENALTY_CONTEXT);
    setImageResolutionBase(DEFAULT_IMAGE_RESOLUTION);
    setImageUrlBase(DEFAULT_IMAGE_URL);
    setGateDelayBase(DEFAULT_GATE_DELAY);
    setGateInfluenceBase(DEFAULT_GATE_INFLUENCE);
    setDisplayColorBase(DEFAULT_DISPLAY_COLOR);
    setCenterCropBase(DEFAULT_CENTER_CROP);
  }, [
    setTextTemperatureBase,
    setTextTopkBase,
    setAudioTemperatureBase,
    setAudioTopkBase,
    setPadMultBase,
    setRepetitionPenaltyBase,
    setRepetitionPenaltyContextBase,
    setImageResolutionBase,
    setImageUrlBase,
    setDisplayColorBase,
    setCenterCropBase,
  ]);

  const setTextTemperature = useCallback((value: number) => {
    if (value <= 1.2 && value >= 0.2) {
      setTextTemperatureBase(value);
    }
  }, []);
  const setTextTopk = useCallback((value: number) => {
    if (value <= 500 && value >= 10) {
      setTextTopkBase(value);
    }
  }, []);
  const setAudioTemperature = useCallback((value: number) => {
    if (value <= 1.2 && value >= 0.2) {
      setAudioTemperatureBase(value);
    }
  }, []);
  const setAudioTopk = useCallback((value: number) => {
    if (value <= 500 && value >= 10) {
      setAudioTopkBase(value);
    }
  }, []);
  const setPadMult = useCallback((value: number) => {
    if (value <= 4 && value >= -4) {
      setPadMultBase(value);
    }
  }, []);
  const setRepetitionPenalty = useCallback((value: number) => {
    if (value <= 2.0 && value >= 1.0) {
      setRepetitionPenaltyBase(value);
    }
  }, []);
  const setRepetitionPenaltyContext = useCallback((value: number) => {
    if (value <= 200 && value >= 0) {
      setRepetitionPenaltyContextBase(value);
    }
  }, []);
  const setImageResolution = useCallback((value: number) => {
    if (value <= 512 && value >= 160) {
      setImageResolutionBase(value);
    }
  }, []);
  const setImageUrl = useCallback((value: string | undefined) => {
    setImageUrlBase(value);
  }, []);
  const setGateDelay = useCallback((value: number) => {
    if (value <= 32 && value >= 0) {
      setGateDelayBase(value);
    }
  }, []);
  const setGateInfluence = useCallback((value: number) => {
    if (value <= 1.0 && value >= 0.0) {
      setGateInfluenceBase(value);
    }
  }, []);
  const setDisplayColor = useCallback((value: boolean) => {
    setDisplayColorBase(value);
  }, []);
  const setCenterCrop = useCallback((value: boolean) => {
    setCenterCropBase(value);
  }, []);
  return {
    textTemperature,
    textTopk,
    audioTemperature,
    audioTopk,
    padMult,
    repetitionPenalty,
    repetitionPenaltyContext,
    imageResolution,
    imageUrl,
    gateDelay,
    gateInfluence,
    displayColor,
    centerCrop,
    setTextTemperature,
    setTextTopk,
    setAudioTemperature,
    setAudioTopk,
    setPadMult,
    setRepetitionPenalty,
    setRepetitionPenaltyContext,
    setImageUrl,
    setImageResolution,
    setGateDelay,
    setGateInfluence,
    setDisplayColor,
    setCenterCrop,
    resetParams,
  }
}