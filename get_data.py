import pathlib
import pandas as pd

from preprocessing import preProcessing


def loadData():
    root = str(pathlib.Path(__file__).parent)
    australian_df = pd.read_csv(root + r'\Australiendeutsch.txt', "utf-8",
                                header=None, names=["Australian"])
    berlin_df = pd.read_csv(root + r'\BerlinerWendekorpus.txt', "utf-8",
                            header=None, names=["Berlin"])
    namibia_df = pd.read_csv(root + r'\DeutschInNamibia.txt', "utf-8",
                             header=None, names=["Namibia"])
    east_germany_df = pd.read_csv(root + r'\EastGermany.txt', "utf-8",
                                  header=None, names=["east_germany"])
    turkish_df = pd.read_csv(root + r'\TurkishReturners.txt', "utf-8",
                             header=None, names=["turkish"])
    great_britain_df = pd.read_csv(root + r'\GreatBritain.txt', "utf-8",
                                   header=None, names=["britain"])
    east_prussia_df = pd.read_csv(root + r'\EastPrussia.txt', "utf-8",
                                  header=None, names=["prussia"])
    german_in_israel_df = pd.read_csv(root + r'\EmigrantGermanInIsrael.txt', "utf-8",
                                      header=None, names=["emigrant_in_israel"])
    viennese_in_israel_df = pd.read_csv(root + r'\VienneseInJerusalem.txt', "utf-8",
                                        header=None, names=["viennese_in_israel"])
    eastern_territories_df = pd.read_csv(root + r'\FormerEasternTerritories.txt', "utf-8",
                                         header=None, names=["eastern_territories"])
    russian_dialects_df = pd.read_csv(root + r'\RussianDialects.txt', "utf-8",
                                      header=None, names=["russian"])
    hamburg_df = pd.read_csv(root + r'\hamburg.txt', "utf-8",
                             header=None, names=["hamburg"])
    return australian_df, berlin_df, namibia_df, east_germany_df, turkish_df, great_britain_df, east_prussia_df, german_in_israel_df, viennese_in_israel_df, eastern_territories_df, russian_dialects_df, hamburg_df


def createDataframe(isHeatmap):
    australian_df, berlin_df, namibia_df, east_germany_df, turkish_df, great_britain_df, east_prussia_df, german_in_israel_df, viennese_in_israel_df, eastern_territories_df, russian_dialects_df, hamburg_df = loadData()
    data_aus, lang_aus = preProcessing(australian_df, "Australian")
    data_berlin, lang_berlin = preProcessing(berlin_df, "Berlin")
    data_namibia, lang_namibia = preProcessing(namibia_df, "Namibia")
    data_east_germany, lang_east_germany = preProcessing(east_germany_df, "east_germany")
    data_turkish, lang_turkish = preProcessing(turkish_df, "turkish")
    data_GB, lang_GB = preProcessing(great_britain_df, "britain")
    data_prussia, lang_prussia = preProcessing(east_prussia_df, "prussia")
    data_german_in_israel, lang_german_in_israel = preProcessing(german_in_israel_df, "emigrant_in_israel")
    data_viennese, lang_viennese = preProcessing(viennese_in_israel_df, "viennese_in_israel")
    data_eastern_territories, lang_eastern_territories = preProcessing(eastern_territories_df, "eastern_territories")
    data_russian_dialects, lang_russian_dialects = preProcessing(russian_dialects_df, "russian")
    data_hamburg, lang_hamburg = preProcessing(hamburg_df, "hamburg")

    df_group1 = pd.DataFrame({"Text": data_berlin + data_hamburg + data_east_germany + data_eastern_territories,
                              "language": lang_berlin + lang_hamburg + lang_east_germany + lang_eastern_territories})
    df_group2 = pd.DataFrame(
        {"Text": data_GB + data_turkish + data_prussia + data_russian_dialects,
         "language": lang_GB + lang_turkish + lang_prussia + lang_russian_dialects})
    df_group3 = pd.DataFrame({"Text": data_aus + data_namibia + data_german_in_israel + data_viennese,
                              "language": lang_aus + lang_namibia + lang_german_in_israel + lang_viennese})
    df_group4 = pd.DataFrame({
                                 "Text": data_aus + data_berlin + data_namibia + data_east_germany + data_turkish + data_GB + data_prussia + data_german_in_israel + data_viennese + data_eastern_territories + data_russian_dialects + data_hamburg,
                                 "language": lang_aus + lang_berlin + lang_namibia + lang_east_germany + lang_turkish + lang_GB + lang_prussia + lang_german_in_israel + lang_viennese + lang_eastern_territories + lang_russian_dialects + lang_hamburg})

    if isHeatmap:
        return [data_aus, data_berlin, data_namibia, data_east_germany, data_turkish, data_GB, data_prussia,
                data_german_in_israel, data_viennese, data_eastern_territories, data_russian_dialects, data_hamburg]
    elif not isHeatmap:
        return df_group1, df_group2, df_group3, df_group4
