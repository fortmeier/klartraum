# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# Downloads the fonts the title text (text_gaussians) is set in to
# <build>/fonts/, together with their licences. All of them are under the SIL
# Open Font License and come from the google/fonts repository at a fixed
# commit; the hashes make the download reproducible. The library finds them as
# "fonts/<file>" through its asset root (see readFile()).

option(KLARTRAUM_DOWNLOAD_FONTS "Download the OFL fonts used for title text" ON)

set(KLARTRAUM_FONTS_COMMIT 47d54ca1ebb14fd864695f1d39dc734f39017c33)
set(KLARTRAUM_FONTS_URL "https://raw.githubusercontent.com/google/fonts/${KLARTRAUM_FONTS_COMMIT}/ofl")
set(KLARTRAUM_FONTS_DIR "${PROJECT_BINARY_DIR}/fonts")

# Name in the build, directory and file in google/fonts (URL-encoded), SHA256
# of the font, SHA256 of its OFL.txt.
set(KLARTRAUM_FONTS
    "SpaceGrotesk|spacegrotesk|SpaceGrotesk%5Bwght%5D.ttf|acad6de1fc93436f5c0f1f4137751ef04f1aea3063e7036535970ffcfbd79f72|564ce565c371c5e5bbf286006565a7c9aa55a9f56e7ca58d56e05d649dd61a72"
    "Geist|geist|Geist%5Bwght%5D.ttf|73894e0448cae90a92b6c2f8732b7bb9acb7b94c418bff559dad4a18e1de9659|1781d2806a07d91c4edf4740b88449fab7d0eadad53f7c351b94cd4d4eb8c00f"
    "Unbounded|unbounded|Unbounded%5Bwght%5D.ttf|323b511be380c8d474ef030686b71aedde501f8d9cd46da558b7c40454372c3f|31e5d4e83955e7103c34570dd49b0570ef490800bd65b42923c0dd02445263b3"
    "Syne|syne|Syne%5Bwght%5D.ttf|ce5ac77142a65cab2248a1a2ebb740b1d4d9c20b52488877d3ff664d1356104a|cc43cdce6f91c57989af8459341c276655e34224e954fa69c2ad700831a742d8"
    "IBMPlexSans|ibmplexsans|IBMPlexSans%5Bwdth%2Cwght%5D.ttf|3b031aa4216174205bd8471f88a49b91f093169e9e87bd5262242bc5967fe2e3|7e6b2818edbd8f6a01ae80641cc8f16a51080d08fb4e532be3a0b6f74adb07da"
    "Fraunces|fraunces|Fraunces%5BSOFT%2CWONK%2Copsz%2Cwght%5D.ttf|177ff6c0f14e5550a3c624247cd1189611d4eb65d000b14944c63d967958abbb|bdf4c22802eaf804f998195871c6b8938aac2ac14b2d78a8bd66a6f1eced833b"
    "JetBrainsMono|jetbrainsmono|JetBrainsMono%5Bwght%5D.ttf|48715a42ec242c21e9f02692891e147d022299a52e48d5e413e1a942193ffeda|b2fe5e8987594e9ffd1d2ca52a2f5d73eb8335243893c5d6254b5ad69269591d"
)

if(KLARTRAUM_DOWNLOAD_FONTS)
    foreach(entry ${KLARTRAUM_FONTS})
        string(REPLACE "|" ";" fields "${entry}")
        list(GET fields 0 name)
        list(GET fields 1 dir)
        list(GET fields 2 file)
        list(GET fields 3 font_hash)
        list(GET fields 4 licence_hash)
        # An existing file with the expected hash is not downloaded again.
        file(DOWNLOAD "${KLARTRAUM_FONTS_URL}/${dir}/${file}" "${KLARTRAUM_FONTS_DIR}/${name}.ttf"
             EXPECTED_HASH SHA256=${font_hash} STATUS font_status)
        file(DOWNLOAD "${KLARTRAUM_FONTS_URL}/${dir}/OFL.txt" "${KLARTRAUM_FONTS_DIR}/${name}.OFL.txt"
             EXPECTED_HASH SHA256=${licence_hash} STATUS licence_status)
        list(GET font_status 0 font_code)
        list(GET licence_status 0 licence_code)
        if(NOT font_code EQUAL 0 OR NOT licence_code EQUAL 0)
            message(WARNING "Could not download the font ${name}; title text tests that need it are skipped.")
        endif()
    endforeach()
endif()
